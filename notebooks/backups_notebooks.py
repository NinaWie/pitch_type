# BALL DETECTION:

# We read the images as grayscale so they are ready for thresholding functions.
#fig = plt.figure(figsize=(18, 16), edgecolor='k')
#plt.imshow(images[0])
#plt.show()


def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    with open(file, 'r') as inf:
        out = json.load(inf)

    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)

    return np.array(liste)

class Node():
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2]
        l = abs(x1-x2)
        w = abs(y1-y2)
        self.area = l*w # (l+w)/float(l*w)
        self.center = [(x1+x2)/2, (y1+y2)/2]
        self.children = []
    def add_child(self, no):
        self.children.append(no)


# PARAMETERS
def detect_ball(folder, joints_array=None, template = "%03d.jpg", min_area = 400, plotting=True):
    # length = len(os.listdir(folder))
    # images = [cv2.imread(folder+template %idx, cv2.IMREAD_GRAYSCALE) for idx in range(length)] #IMG_TEMPLATE.format(idx), )

    cap = cv2.VideoCapture(folder)
    images=[]
    start = time.time()

    candidates_per_frame = []
    candidate_values = []
    whiteness_values = []
    location = []
    frame_indizes = []
    balls = []
    t=0
    frame_before_close_wrist = False
    while t<150:
        ret, frame = cap.read()
        candidates_per_frame.append([])
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # for t in range(1, length-1):
        if t<2:
            images.append(frame)
            t+=1
            continue

        im_tm1 = images[0]
        im_t = images[1]
        im_tp1 = frame
        delta_plus = cv2.absdiff(im_t, im_tm1)
        delta_0 = cv2.absdiff(im_tp1, im_tm1)
        delta_minus = cv2.absdiff(im_t,im_tp1)
        sp = cv2.meanStdDev(delta_plus)
        sm = cv2.meanStdDev(delta_minus)
        s0 = cv2.meanStdDev(delta_0)
        # print("E(d+):", sp, "\nE(d-):", sm, "\nE(d0):", s0)

        th = [
            sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
            sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
            s0[0][0, 0] + 3 * math.sqrt(s0[1][0, 0]),
        ]

        #print("Thresholds:", th)
        """
        # NAIVE THRESHOLD
        start = time.time()
        dbp = threshold(delta_plus, th[0])
        dbm = threshold(delta_minus, th[1])
        db0 = threshold(delta_0, th[2], invert=True)

        detect_naive = combine(dbp, dbm, db0)
        naive_time = (time.time() - start) * 1000
        """

        # OPENCV THRESHOLD

        ret, dbp = cv2.threshold(delta_plus, th[0], 255, cv2.THRESH_BINARY)
        ret, dbm = cv2.threshold(delta_minus, th[1], 255, cv2.THRESH_BINARY)
        ret, db0 = cv2.threshold(delta_0, th[2], 255, cv2.THRESH_BINARY)

        detect = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(dbp, dbm),
                        cv2.bitwise_not(db0)))

        # CONNECTED BOX
        # The original `detect` image was suitable for display, but it is "inverted" and not suitable
        # for component detection; we need to invert it first.
        start = time.time()
        nd = cv2.bitwise_not(detect)
        # only stats is used, not num, labels, centroids
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)
        # We set an arbitrary threshold to screen out smaller "components"
        # which may result simply from noise, or moving leaves, and other
        # elements not of interest.


        d = detect.copy()
        candidates = list()
        for stat in stats[1:]:
            area = stat[cv2.CC_STAT_AREA]
            if area < min_area:
                continue # Skip small objects (noise)

            lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
            rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
            bottomLeftCornerOfText = (lt[0], lt[1] - 15)

            candidates.append((lt, rb, area))

        #plt.figure(figsize=(10, 10), edgecolor='k')

        # MOTION MODEL

        # Thinning threshold.
        psi = 0.7
        # Area matching threshold.
        gamma = 20 #0.4 changed to see all candidates
        area_max = 0

        fom_detected = False
        start = time.time()
        col = len(candidates)
        #print("number of candidates", col)
        sub_regions = list()

        #if len(candidates)>0:
         #   wrist_position=joints_array[t-1, [1, 2, 4, 5],:] # ellbows and wrists
            #box = np.sqrt(min_area)
            #patch_right = [wrist_position[0,0]-box,wrist_position[0,0]+box, wrist_position[0,1]-box, wrist_position[0,1+box]
        index = []
        close_to_wrist=False
        for i, candidate in enumerate(candidates):

            # The first two elements of each `candidate` tuple are
            # the opposing corners of the bounding box.
            x1, y1 = candidate[0]
            x2, y2 = candidate[1]
            center = [(x1+x2)/2, (y1+y2)/2]

            no = Node(x1, y1, x2, y2)
            candidates_per_frame[-1].append(no)
            if t>0 and candidates_per_frame[-2]!=[]:
                for nodes in candidates_per_frame[-2]:
                    nodes.add_child(no)
                    print("previous detection", nodes.bbox, "gets child", no.bbox)
            """
            # CLOSE WRIST IDEA
            if frame_before_close_wrist:
                bbox = [np.min(joints_array[t-1, :, 0]), np.max(joints_array[t-1,:, 0]),
                       np.min(joints_array[t-1, :, 1]), np.max(joints_array[t-1, :, 1])]
                # if not near somewhere the body
                if not (bbox[0]<center[0]<bbox[1] and bbox[2]<center[1]<bbox[3]):
                    #print(bbox, center)
                    print("BALL DETECTED, ", center)

            if close_to_wrist==False:
                # if center of detection box is close enough (2*length of min_area), close_to_wrist = True
                distances = [np.linalg.norm(center-wrist_position[i]) for i in range(len(wrist_position))]
                if np.any(distances<3*np.sqrt(min_area)):# or np.linalg.norm(center-wrist_position[1])<2*np.sqrt(min_area):
                    close_to_wrist = True
                    print("close to wrist with center:", center)
            """
            # We had placed the candidate's area in the third element of the tuple.
            actual_area = candidate[2]

            # For each candidate, estimate the "radius" using a distance transform.
            # The transform is computed on the (small) bounding rectangle.
            cand = nd[y1:y2, x1:x2]
            dt = cv2.distanceTransform(cand, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
            radius = np.amax(dt)

            # "Thinning" of pixels "close to the center" to estimate a
            # potential FOM path.
            ret, Pt = cv2.threshold(dt, psi * radius, 255, cv2.THRESH_BINARY)

            # TODO: compute actual path lenght, using best-fit straight line
            #   along the "thinned" path.
            # For now, we estimate it as the max possible lenght in the bounding box, its diagonal.
            w = x2 - x1
            h = y2 - y1
            path_len = math.sqrt(w * w + h * h)
            expected_area = radius * (2 * path_len + math.pi * radius)

            area_ratio = abs(actual_area / expected_area - 1)
            #print(area_ratio)

            location.append(center)
            index.append(i)
            #area = candidates[i][2]
            #candidate_values.append(area_ratio)
            #patch = im_t[y1:y2, x1:x2]
            #whiteness_values.append(np.mean(patch))
            frame_indizes.append(t-1)

        ### BALL DETECTION:
        if t>1 and candidates_per_frame[-2]!=[]:
            area_diff=[]
            nodes = []
            for cand in candidates_per_frame[-2]:
                for c in cand.children:
                    area_diff.append(abs(cand.area- c.area))
                    nodes.append(c)
            print(t, area_diff)
            if area_diff!=[] and np.min(area_diff)<min_area:
                if len(balls)>1 and len(np.where(np.array(area_diff)<min_area)[0])>1:
                    d = []
                    for j, n in enumerate(nodes):
                        if area_diff[j]<min_area:
                            p1 =  np.array([(balls[0][0]+balls[0][2])/2, (balls[0][1]+balls[0][3])/2])
                            p2 =  np.array([(balls[1][0]+balls[1][2])/2, (balls[1][1]+balls[1][3])/2])
                            p3 =  np.array([(n.bbox[0]+n.bbox[2])/2, (n.bbox[1]+n.bbox[3])/2])
                            d.append(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
                        else: d.append(np.inf)
                    balls.append(nodes[np.argmin(d)].bbox)
                else:
                    balls.append(nodes[np.argmin(area_diff)].bbox) # ändern statt d area_diff
                # print(balls)
            else:
                balls = []
        else:
            balls = []
        if len(balls)==5:
            print("release frame", i-5)
            break
        ###

        if plotting and len(balls)>0: #len(candidates)>0:
            print("DETECTED", t-1, whiteness_values[-1], candidate_values[-1])
            plt.figure(figsize=(10, 10), edgecolor='r')
            # print(candidates[fom])
            img = np.tile(np.expand_dims(im_t.copy(), axis = 2), (1,1,3))
            #print(img.shape)
            #for jo in wrist_position:
             #   cv2.circle(img, (int(jo[0]), int(jo[1])), 8, [255,0,0], thickness=-1)

            # for fom in index: # einfuegen falls alles plotten
            cv2.rectangle(img,tuple(balls[-1][:2]), tuple(balls[-1][2:]), [255,0,0], 4)
                              # candidates[fom][0], candidates[fom][1],[255,0,0], 4)
            plt.imshow(img, 'gray')
            plt.title("Detected FOM".format(t))
            plt.show()
        """
        if close_to_wrist:
            frame_before_close_wrist = True
        else:
            frame_before_close_wrist = False
        """
        t+=1
        images[0] = images[1].copy()
        images[1] = frame
    print("time for %s frames without plotting"%t, (time.time() - start) * 1000)

    return frame_indizes, location, candidates_per_frame

# 40mph_1us_1.2f_170fps_40m_sun # 40mph_10us_6f_100fps_40m_cloudy # 40mph_10us_11f_100fps_noisy.avi
frame_indizes, location, candidates_per_frame = detect_ball("/Volumes/Nina Backup/Nina's Pitch/40mph_10us_6f_100fps_40m_cloudy.avi")
sys.exit()

example = "#10 Matt Glomb" #26 RHP Tim Willites" (willites camera moves) # #00 RHP Devin Smith # #10 Matt Glomb
BASE = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/batter/"+example+".mp4" #"data/Matt_Blais/" # für batter: pic davor, 03 streichen und (int(idx+1))
joints_path = "/Volumes/Nina Backup/high_quality_outputs/"+example+".json"

#BASE = "/Volumes/Nina Backup/CENTERFIELD/4f7477b1-d129-4ff7-a83f-ad322de63b24.mp4"
#joints_path = "/Volumes/Nina Backup/outputs/new_videos/cf/490770_4f7477b1-d129-4ff7-a83f-ad322de63b24_pitcher.json"
joints = from_json(joints_path)[:,:12,:]
print(joints.shape)
frame_indizes, location, candidates_per_frame = detect_ball(BASE, joints_array = joints, min_area=400)

### ONLY DETECTION

balls = []
min_area = 400
for i in range(len(candidates_per_frame)):
    if candidates_per_frame[i]!=[]:
        area_diff=[]
        nodes = []
        for cand in candidates_per_frame[i]:
            for c in cand.children:
                area_diff.append(abs(cand.area- c.area))
                nodes.append(c)
        print(i, area_diff)
        if area_diff!=[] and np.min(area_diff)<min_area:
            if len(balls)>1 and len(np.where(np.array(area_diff)<min_area)[0])>1:
                d = []
                for j, n in enumerate(nodes):
                    if area_diff[j]<min_area:
                        p1 =  np.array([(balls[0][0]+balls[0][2])/2, (balls[0][1]+balls[0][3])/2])
                        p2 =  np.array([(balls[1][0]+balls[1][2])/2, (balls[1][1]+balls[1][3])/2])
                        p3 =  np.array([(n.bbox[0]+n.bbox[2])/2, (n.bbox[1]+n.bbox[3])/2])
                        d.append(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
                    else: d.append(np.inf)
                balls.append(nodes[np.argmin(d)].bbox)
            else:
                balls.append(nodes[np.argmin(area_diff)].bbox) # ändern statt d area_diff
            # print(balls)
        else:
            balls = []
    else:
        balls = []
    if len(balls)==5:
        print("release frame", i-5)
        break
print(balls)
balls = np.array(balls)
plt.scatter(balls[:,0], balls[:,1])
plt.ylim(1000, 0)
plt.xlim(0,2000)
plt.show()


### ALL DIFFERENT APPROACHES

#print(candidate_values)
#print(np.argmax(candidate_values))
#print(np.argsort(candidate_values))
#print(frame_indizes)
#print(whiteness_values)
#print(delta_plus.shape, delta_0.shape, delta_minus.shape)
location = np.array(location)
plt.scatter(location[:,0], location[:,1])
plt.show()

def find_consecutive_frame():
    count =0
    frame = 0
    results=[]
    for detection in range(len(frame_indizes)):
        same_frame = frame_indizes[detection]==frame
        frame = frame_indizes[detection]
        bbox = [np.min(joints[frame, :, 0]), np.max(joints[frame,:, 0]),
                       np.min(joints[frame, :, 1]), np.max(joints[frame, :, 1])]
        if bbox[0]<location[detection, 0]<bbox[1] and bbox[2]<location[detection, 1]<bbox[3]:
            if not same_frame:
                count = 0
            continue
        #distance = min([np.linalg.norm(location[detection]-joints[frame, i]) for i in range(12)])
        if np.all(joints[frame, :, 1]>location[detection,1]):
            count+=1
        elif not same_frame:
            count = 0

        if count==3:
            print("found three points after another", frame)
            results.append(frame)
    return results

#res = find_consecutive_frame()
#print("results of find_consecutive_frame", res)

# find highest detections
def consecutive_highest_candidates(location, frame_indizes):
    a = np.argsort(location[:,1])
    x = np.array(frame_indizes)[a]
    result = np.median(x[:5])
    print("results of median of lowest frames", result)

    # find consecutive frames in highest ones
    found=False
    nr = 10
    while not found:
        sort_x = np.sort(x[:nr])
        print(sort_x)
        count=0
        for i in range(1, len(sort_x)):
            if abs(sort_x[i]-sort_x[i-1])==1:
                count+=1
            else:
                if count>2:
                    found=True
                    print("found consecutive frame", sort_x[i-2])
                    result = sort_x[i-2]
                    break
                count=0
        print("just argmin", frame_indizes[np.argmin(location[:,1])])
        nr+=10
    return result

# look for points which are on a line, such that cross product is zero

plt.scatter(frame_indizes, location[:,1])
plt.plot(joints[:, 2,1], color = "green")
plt.show()
plt.scatter(frame_indizes, np.gradient(location[:,1]))
#plt.plot(joints[:, 2,1], color = "green")
plt.show()
plt.scatter(frame_indizes, location[:,0])
plt.plot(joints[:, 2,0], color = "green")
plt.show()
plt.scatter(location[:,0], location[:,1])
plt.show()



### COLOR VIDEO USW


import codecs
import json

def visualize(data, nr):
    for i in range(nr):
        print(np.any(np.isnan(data[i])))
        plt.figure()
        plt.plot(data[i])
        plt.show()

def normalize(data):
    """
    normalizes across frames - axix to zero mean and standard deviation
    """
    M,N, nr_joints,_ = data.shape
    means = np.mean(data, axis = 1)
    std = np.std(data, axis = 1)
    res = np.asarray([(data[:,i]-means)/(std+0.000001) for i in range(len(data[0]))])
    data_new = np.swapaxes(res, 0,1)
    return data_new

def normalize01(data):
    maxi = np.amax(data, axis=1)
    mini = np.amin(data, axis = 1)
    res = np.asarray([(data[:,i]-mini)/(maxi-mini) for i in range(len(data[0]))])
    data_new = np.swapaxes(res, 0,1)
    return data_new

def normalize_whole(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/std

from numpy import random, sqrt
from scipy import stats
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)

def smooth(x,window_len=6,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

def do_pca(data, nr_components):
    from sklearn.decomposition import PCA
    M, frames, joi, xy = data.shape
    res = data.reshape(M,frames,joi*xy)
    res = res.reshape(M*frames, joi*xy)
    #print(data[1,3])
    #print(res[170])
    print(res.shape)
    pca = PCA(n_components=nr_components)
    new_data = pca.fit_transform(res)
    # new_data = pca.transform(res)
    print(new_data.shape)
    new = new_data.reshape(M,frames, nr_components)
    return new

def pearson_distance(x,y):
    corrs = []
    assert(x.shape==y.shape)
    if len(x.shape)>2:
        for j in range(12):
            for xy in range(2):
                a,_ = stats.pearsonr(x[:,j,xy], y[:,j,xy])
                if not np.isnan(a):
                    corrs.append(a)
        if np.isnan(np.mean(corrs)):
            print(corrs)
        return np.mean(corrs)
    elif len(x.shape)==2:
        for j in range(x.shape[1]):
            a,_ = stats.pearsonr(x[:,j], y[:,j])
            if not np.isnan(a):
                corrs.append(a)
        return np.mean(corrs)
    elif len(x.shape)==0:
        a,_ = stats.pearsonr(x, y)
        return a
    else:
        print("INVALID DIMENSION!")
        raise TypeError("invalid dimension")

def dist_dtak(play, template):
    result = []
    xy = 1
    for joint in range(len(play[0])):
        x = play[:,joint,xy]
        y = template[:,joint,xy]
        dist, cost, acc, path = fastdtw(x, y, dist=lambda x, y: np.linalg.norm(x - y))

        path_length = len(path[0])

        aligned_x = x[path[0]]
        aligned_y = y[path[1]]
        diff = np.sum(np.absolute(aligned_x-aligned_y))
        result.append(diff+((path_length-200)/2))
    # print(result)
    return int(np.median(result))


def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=data[random.permutation(len(data))[:num_clust]]
    # print(centroids.shape)
    for n in range(num_iter):
        print(n)
        if n>0:
            old_assignments = assignments

        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist= float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                #if LB_Keogh(i,j,5)<min_dist:
                 #   cur_dist=DTWDistance(i,j,w)
                    # if cur_dist<min_dist:
                cur_dist = dist_dtak(i,j) # pearson_distance(i,j)
                #print(cur_dist)
                if cur_dist<min_dist:
                    min_dist=cur_dist
                    closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[ind]

        #diffs = []
        if n>0:
            if old_assignments==assignments:
                break

        #recalculate centroids of clusters
        for key in assignments:
            #old_centroid = centroids[key].copy()
            clust_sum = 0 # data[]
            for k in assignments[key]:
                clust_sum += data[k]

            centroids[key]= clust_sum/float(len(assignments[key]))
            #diffs.append(LB_Keogh(old_centroid, centroids[key], 1))
        #print(diffs)

    return centroids, assignments

import pandas as pd
from scipy import ndimage
from os import listdir
import numpy as np
import matplotlib.pylab as plt

def get_data_pitchtype():
    csv = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/csvs/csv_gameplay.csv", delimiter = ";")

    path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/pitcher/"
    joints_array_pitcher = []
    files = []
    pitchtype = []
    #release = []

    #with open(path+"release_frames Kopie", "r") as infile:
     #   release_frame = json.load(infile)

    for fi in listdir(path):
        if fi[-5:]==".json":
            line = csv[csv["play_id"]==fi[:-5]]
            try:
                pitch  = line["Pitch Type"].values[0]
    #          release.append(release_frame[fi[:-5]])
            except IndexError:
                continue


            obj_text = codecs.open(path+fi, encoding='utf-8').read()

            arr = json.loads(obj_text)
            if np.all(np.array(arr)==0):
                continue

            joints_array_pitcher.append(np.array(arr)[60:110,:12,:])
            pitchtype.append(pitch)
            files.append(fi[:-5])

    joints_array_pitcher = np.array(joints_array_pitcher)
    joints_array_pitcher = ndimage.filters.gaussian_filter1d(joints_array_pitcher, axis =1, sigma = 2)
    print(joints_array_pitcher.shape)
    # print(hit_into)
    assert(len(pitchtype)==len(joints_array_pitcher))

    return normalize(joints_array_pitcher), files, pitchtype

def get_data_run_norun():

    csv = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/csvs/csv_gameplay.csv", delimiter = ";")

    path="/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/different_batters/batter/"
    joints_array_batter = []
    files = []
    hit_into = []
    play_outcome = []
    #release = []

    #with open(path+"release_frames Kopie", "r") as infile:
     #   release_frame = json.load(infile)

    for fi in listdir(path):
        if fi[-5:]==".json":
            line = csv[csv["play_id"]==fi[:-5]]
            try:
                hit  = line["Hit into play?"].values[0]
                out = line["Play Outcome"].values[0]
    #          release.append(release_frame[fi[:-5]])
            except IndexError:
                continue


            obj_text = codecs.open(path+fi, encoding='utf-8').read()

            arr = json.loads(obj_text)
            if np.all(np.array(arr)==0):
                continue

            if "Foul" in out or "Swinging strike" in out:
                lab = "hit"
            elif "Ball/Pitcher" in out or "Called strike" in out:
                lab = "nothing"
            elif "Hit into play" in out:
                lab = "run"
            else:
                continue
            joints_array_batter.append(arr)
            hit_into.append(hit)
            play_outcome.append(lab)
            files.append(fi[:-5])

    joints_array_batter = np.array(joints_array_batter)[:,:,:12,:]
    joints_array_batter = ndimage.filters.gaussian_filter1d(joints_array_batter, axis =1, sigma = 5)
    print(joints_array_batter.shape)
    # print(hit_into)
    assert(len(hit_into)==len(joints_array_batter))
    assert(len(hit_into)==len(play_outcome))
    # print(release)
    return normalize(joints_array_batter), files, hit_into, play_outcome

def get_data_pitcher_batter():
    path_batter = "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/video_to_pitchtype_directly/out_testing_batter/"
    path_pitcher = "/Users/ninawiedemann/Desktop/UNI/Praktikum/ALL/Pose_Estimation/out_joints/"

    batter = np.zeros((15, 167, 12, 2))
    i=0
    for fi in listdir(path_batter):
        if fi[-5:]==".json":
            obj_text = codecs.open(path_batter+fi, encoding='utf-8').read()
            arr = json.loads(obj_text)
            batter[i, :len(arr)] = np.array(arr)[:,:12,:]
            i+=1
            if i>14:
                break
    pitcher = np.zeros((15, 167, 12, 2))
    for i, fi in enumerate(listdir(path_pitcher)):
        obj_text = codecs.open(path_pitcher+fi, encoding='utf-8').read()
        pitcher[i] = np.array(json.loads(obj_text))[:,:12,:]
        if i>13:
            break

    data = []

    for i in range(15):
        data.append(batter[i])
        data.append(pitcher[i])
    # visualize(data)
    return normalize(np.array(data))



 # player localization
import cv2
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import json
from scipy.signal import argrelextrema
from scipy import ndimage
import ast
from scipy.spatial.distance import cdist
import pandas as pd
from os import listdir

player_list = [0]
joints_for_cdist = np.arange(0,18,1)
important_joints = [0,1,6,7,8,9,10,11]
def color_video(json_array, vid_file, start = 0, end = 300, printing =None, plotting=True):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 170, 255],
              [0, 0, 0],
          [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255],
              [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255], \
          [170, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors_string = ["blue", "green", "red", "tuerkis", "pink", "yellow", "orange", "black", "purple"]
    nr_joints =12
    #print(json_array.shape)
    #writer = cv2.VideoWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi",cv2.VideoWriter_fourcc(*"XVID") , 20, (500,800))

    #writer = io.FFmpegWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi", (10,800,500,3))
    #writer.open()
    video_capture = cv2.VideoCapture(vid_file)
    if start!=0:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    arr = [] #np.zeros((100,800,500,3))

    # fig = plt.figure(figsize=(5, 15)) # for subplots
    for k in range(start, end):
        #print(k)
        if printing!=None:
            #print("dist_min",  "ratio_min")
            #print(colors_string[printing[k][0]], colors_string[printing[k][1]])
            print(printing[k])
        ret, frame = video_capture.read()
        if frame is None:
            break
        if len(json_array[k].shape)==2:
            all_peaks = np.reshape(json_array[k], (12, 1,2))
        else:
            all_peaks = json_array[k]
        #print(all_peaks.shape)
        canvas = frame #[top_b:bottom_b, left_b:right_b] # cv2.imread(f) # B,G,R order
        oriImg = canvas.copy()

        for i in range(len(all_peaks)):
            #print("person", all_peaks[i])
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, (int(all_peaks[i,j,1]),int(all_peaks[i,j,0])) , 8, colors[i], thickness=-1)

        to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        arr.append(to_plot[:,:,[2,1,0]])
        if plotting:
            width, height, _ = frame.shape
            print(width,height)
            plt.figure(figsize = (width/float(100), height/float(100)))
            plt.imshow(to_plot[:,:,[2,1,0]])
            #fig = matplotlib.pyplot.gcf()
            #fig.set_size_inches(12, 12)
            plt.show()

    """
    # for subplots
    ax = fig.add_subplot(end-start,1, k-start+1)
        plt.imshow(to_plot[:,:,[2,1,0]])
        plt.title("frame "+str(k))

    plt.tight_layout()
    plt.show()
    """
    arr = np.array(arr)
    return arr



path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/"
outputs = path + "handle_one/"
argmin_list = []
with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/center_dics.json", "r") as infile:
    dictionary = json.load(infile)
# df = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one/"+name+".csv", dtype = {"Frame": np.int32, "Batter": np.ndarray})
for json_file in ["#5 RHP Matt Blais (3).json"]:
    name = json_file[:-5]
    if json_file[-5:]!= ".json" or name+".mp4" in outputs:
        continue
    print(name)
    with open(path +"test_multiplier"+".json", "r") as infile:
        handle_one_arr = json.load(infile)

    if name+".mp4" in listdir(path+"batter"):
        f = path+"batter/"+name+".mp4"
    else:
        f = path+"pitcher/"+name+".mp4"
    print(len(handle_one_arr))


    for i in range(len(handle_one_arr)):
        handle_one_arr[i]=np.array(handle_one_arr[i])

    print(handle_one_arr[0].shape)

    df_handle = [handle_one_arr, handle_one_arr.copy()]

    #center = [dictionary[name][1],dictionary[name][0]]
    #new_df = np.array(df_coordinates(df_handle, np.array([center]), player_list)[1])
    #print("out_df", np.array(new_df).shape)

    arr = color_video(handle_one_arr, f, start = 0, end = 4 )#len(handle_one_arr)) #, printing = argmin_list)
    # skvideo.io.vwrite("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one_bsp.mp4", arr)
    # arr2 = color_video(new_df[:,:12,:], f, start = 0, end =len(new_df), printing = None, plotting=True)
    #io.vwrite("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one/"+name+".mp4", arr2)






# PLAYER LOCALIZATION

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import json
from scipy.signal import argrelextrema
from scipy import ndimage
import ast
from scipy.spatial.distance import cdist
from skvideo import io
from os import listdir
import pandas as pd
/Users/ninawiedemann/anaconda/lib/python3.5/site-packages/skvideo/__init__.py:356: UserWarning: avconv/avprobe not found in path:
  warnings.warn("avconv/avprobe not found in path: " + str(path), UserWarning)
In [122]:

player_list = [0]
joints_for_cdist = np.arange(0,18,1)
important_joints = [0,3,6,7,8,9,10,11]
def color_video(json_array, vid_file, start = 0, cut_frame = True, end = 300, point = 8, printing =None, plotting=True):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 170, 255],
              [0, 0, 0],
          [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255],
              [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255], \
          [170, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors_string = ["blue", "green", "red", "tuerkis", "pink", "yellow", "orange", "black", "purple"]
    nr_joints =12
    #print(json_array.shape)
    #writer = cv2.VideoWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi",cv2.VideoWriter_fourcc(*"XVID") , 20, (500,800))
​
    #writer = io.FFmpegWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi", (10,800,500,3))
    #writer.open()
    video_capture = cv2.VideoCapture(vid_file)
    print(vid_file)
    if start!=0:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    arr = [] #np.zeros((100,800,500,3))
​

    # fig = plt.figure(figsize=(5, 15)) # for subplots
    for k in range(start, end):
        #print(k)
        if printing!=None:
            #print("dist_min",  "ratio_min")
            #print(colors_string[printing[k][0]], colors_string[printing[k][1]])
            print(printing[k])
        ret, frame = video_capture.read()
        if frame is None:
            print("end", k)
            break
        if len(np.array(json_array[k]).shape)==2:
            all_peaks = np.reshape(np.array(json_array[k]), (12, 1,2))
        else:
            all_peaks = np.array(json_array[k])
        #print(all_peaks.shape)

        if cut_frame:
            canvas = frame[top_b:bottom_b, left_b:right_b] # cv2.imread(f) # B,G,R order
        else:
            canvas = frame
        oriImg = canvas.copy()

        for i in range(len(all_peaks)):
            #print("person", all_peaks[i])
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, (int(all_peaks[i,j,0]),int(all_peaks[i,j,1])) , point, colors[i], thickness=-1)
​
        to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        arr.append(to_plot[:,:,[2,1,0]])
        if plotting:
            plt.imshow(to_plot[:,:,[2,1,0]])
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(12, 12)
            plt.show()
​
    """
    # for subplots
    ax = fig.add_subplot(end-start,1, k-start+1)
        plt.imshow(to_plot[:,:,[2,1,0]])
        plt.title("frame "+str(k))

    plt.tight_layout()
    plt.show()
    """
    arr = np.array(arr)
    return arr
​
def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "left_shoulder", "right_elbow", "right_wrist","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck ",
            "right_eye", "right_ear","left_eye", "left_ear"]
    with open(file, 'r') as inf:
        out = json.load(inf)
​
    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)
    return np.array(liste)
​
def smooth_estelle(x,window_len=6,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    #print(len(x))
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
​
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

​
    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

​
    s = np.r_[x[window_len//2-1:0:-1],x,x[-1:-window_len//2:-1],x[-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    #print(w)

    y=np.convolve(w/w.sum(),x,mode='same')
    #print(len(y))
    return y
​
def kalmann(sequence):
    # intial parameters
    n_iter = len(sequence)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    z = sequence #np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
​
    Q = 1e-5 # process variance
​
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
​
    R = 0.1**3 # estimate of measurement variance, change to see effect
​
    # intial guesses
    xhat[0] = sequence[0]
    P[0] = 1.0
​
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
​
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat
​
def player_localization_old(df,frame,player,old_array, body_dist):
    #player2=player+'_player'
    dist=[]
    ratios = []
    zerrow2=np.where(old_array[:,0]!=0)[0]
    for i in range(np.asarray(df[player][frame]).shape[0]):
        zerrow1=np.where(np.asarray(df[player][frame])[i,:,0]!=0)[0]
        zerrow_all =np.intersect1d(zerrow1,zerrow2) # assume unique argument for speedup?
        zerrow = np.intersect1d(zerrow_all, important_joints)
        # print("leng", len(zerrow), zerrow)
​
        if len(zerrow)<2:
            dist.append(np.inf)
            ratios.append(np.inf)
            continue
​
        dist.append(np.linalg.norm(np.asarray(df[player][frame])[i,zerrow,:] - old_array[zerrow])/len(zerrow))
​
        p = df[player][frame][i]
        player_dist = cdist(p,p)
​
        def cut_nonzero(cdi, nonzero):
            cdi = (cdi[nonzero])
            cdi = np.swapaxes(cdi,0,1)
            cdi = cdi[nonzero]
            cdi = np.swapaxes(cdi,0,1)
            return cdi
​
        ratios.append(np.linalg.norm(cut_nonzero(body_dist, zerrow) - cut_nonzero(player_dist, zerrow))/len(zerrow))
​
​
    #print df[player][frame]
​
    if len(dist)==0:
        df[1][frame]=[[0,0] for i in range(18)]
        #print("ungleich", frame, np.argmin(ratios), np.argmin(dist))
    elif len(dist)==1:
        df[1][frame]= df[player][frame][0]
    else:
        #df[1][frame]=df[player][frame][np.argmin(smallest_dist[0])]
        smallest_dist = np.argsort(dist)
        argmin_list.append(smallest_dist) #[np.argmin(dist), np.argmin(ratios)])
        if dist[smallest_dist[1]] > 2*dist[smallest_dist[0]]:
            df[1][frame]=df[player][frame][smallest_dist[0]]
        else:
            if ratios[smallest_dist[0]]<ratios[smallest_dist[1]]: #smallest_dist[0]== np.argmin(ratios) or smallest_dist[1]==np.argmin(ratios):
                df[1][frame]=df[player][frame][smallest_dist[0]]
            else:
                df[1][frame]=[[0,0] for i in range(18)]
    array_stored=np.asarray(df[1][frame])
    array_stored[np.where(array_stored==0)]=old_array[np.where(array_stored==0)]
​
    joint_arr_cdist = np.array(array_stored)[joints_for_cdist]
    new_body_dist = cdist(joint_arr_cdist, joint_arr_cdist)
    old_array=array_stored
    return df, old_array, new_body_dist
​
def color_box(vid, bbox, color = "red"):
    video_capture = cv2.VideoCapture(vid)
    ret, frame = video_capture.read()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame[top_b:bottom_b, left_b:right_b], aspect='equal')
    ax.add_patch(
    plt.Rectangle((int(bbox[0]), int(bbox[2])),
                  int(bbox[1]-bbox[0]), int(bbox[3]-bbox[2]), fill=False,
                  edgecolor=color, linewidth=3.5)
    )
    plt.show()

​
def overlap(A, B):
    #print(A, B)
    if (A[0] > B[1]) or (A[1] < B[0]):
        #print(A[0], ">", B[1], "or", A[1], "<", B[0])
        return 0
    if (A[2] > B[3]) or (A[3] < B[2]):
        #print(A[2], ">", B[3], "or", A[3], "<", B[2])
        return 0
    I = [max(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), min(A[3], B[3])]
    # color_box("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/#5 RHP Matt Blais (4).mp4", I, )
    color_box(video_color_box, I,color = "green")

    Aarea = abs((A[0]-A[1])*(A[2]-A[3]))
    Barea = abs((B[0]-B[1])*(B[2]-B[3]))
    Iarea = abs((I[0]-I[1])*(I[2]-I[3]))

    #print(Aarea, Barea, Iarea)
    return Iarea/(Aarea+Barea-Iarea)
​
# IOU
def player_localization(df,frame,player,old_array, body_dist):
    zerrow2=np.where(old_array[:,0]!=0)[0]
    joints_for_bbox = np.intersect1d(zerrow2, important_joints)
    #print("Frame", frame, "old array", old_array[joints_for_bbox])
    old_arr_bbox = [np.min(old_array[joints_for_bbox, 0]), np.max(old_array[joints_for_bbox, 0]),
                   np.min(old_array[joints_for_bbox, 1]), np.max(old_array[joints_for_bbox, 1])]
    #print(old_arr_bbox)
    intersections = []
    boxes = []
    color_box(video_color_box, old_arr_bbox)

    for i in range(np.asarray(df[player][frame]).shape[0]):
        player_array = df[player][frame][i]

        zerrow1=np.where(np.asarray(df[player][frame])[i,:,0]!=0)[0]
        zerrow_all =np.intersect1d(zerrow1,zerrow2) # assume unique argument for speedup?
        zerrow = np.intersect1d(zerrow_all, important_joints)

        if len(zerrow)<2:
            intersections.append(0)
            continue

        joints_for_bbox = np.intersect1d(zerrow1, important_joints)
        player_arr_bbox = [np.min(player_array[joints_for_bbox, 0]), np.max(player_array[joints_for_bbox, 0]),
                        np.min(player_array[joints_for_bbox, 1]), np.max(player_array[joints_for_bbox, 1])]
        print(player_arr_bbox, df[player][frame][i])
        # print(player_arr_bbox)
        intersections.append(overlap(player_arr_bbox, old_arr_bbox))
        color_box(video_color_box, player_arr_bbox, color = "blue")
        print(i, intersections[-1], "frame", frame)
        boxes.append(player_arr_bbox)
        #if intersections[-1] > 0:
            #print("overlap", frame, i)
            #intersect = bb_intersection_over_union(old_arr_bbox, player_arr_bbox)
            #print("inter", intersect)
            #color_box("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/#5 RHP Matt Blais (4).mp4", player_arr_bbox, color = "blue")
        #else:
            #print("no overlap")
    if not np.any(np.array(intersections)>0.1):
        #print("intersections", intersections)
        #color_box("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/#5 RHP Matt Blais (4).mp4", old_arr_bbox, color = "blue")
        #for j in boxes:
         #   print("overlap failed",overlap(j, old_arr_bbox))
             #color_box("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/#5 RHP Matt Blais (4).mp4", j, color = "green")
        df[1][frame]=[[0,0] for i in range(18)]
    else:
        df[1][frame]= df[player][frame][np.argmax(intersections)]

    array_stored=np.asarray(df[1][frame])
    # print("new_array", array_stored)
    array_stored[np.where(array_stored==0)]=old_array[np.where(array_stored==0)]
​
    joint_arr_cdist = np.array(array_stored)[joints_for_cdist]
    new_body_dist = cdist(joint_arr_cdist, joint_arr_cdist)
    old_array=array_stored
    return df, old_array, new_body_dist
    #artificial_bbox = np.array(old_arr_bbox)+50
    #print(overlap(artificial_bbox, old_arr_bbox))

video_color_box = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/490987-79a32c9c-d4dc-4a63-aae9-804f697eb56d.mp4"
bottom_b = 269
left_b =499
right_b= 703
top_b = 103
​
def df_coordinates(df,centerd, player_list, interpolate = True):
    #df.sort_values(by='Frame',ascending=1,inplace=True)
    #df.reset_index(inplace=True,drop=True)
    for player in player_list:
        #df[player+'_player']=df[player].copy()
        #player2=player+'_player'
        center=centerd[player]
        old_norm=10000
        indices=[6,9]
        #print df[player][0]
        for person in range(len(df[player][0])):
            hips=np.asarray(df[player][0][person])[indices]
​
            hips=hips[np.sum(hips,axis=1)!=0]
            mean_hips=np.mean(hips,axis=0)
            #print(mean_hips, center)
​
​
            norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
            if norm<old_norm:
​
                loc=person
                old_norm=norm
        argmin_list.append([loc, loc])
        df[1][0]=df[player][0][loc]
        globals()['old_array_%s'%player]=np.asarray(df[player][0][loc])
        joint_arr_cdist = np.array(df[player][0][loc])
        print("joints_arr_cdist", np.array(joint_arr_cdist).shape)
        globals()['cdist_%s'%player] = cdist(joint_arr_cdist, joint_arr_cdist)
​
    for frame in range(1,len(df[0])):
        for player in player_list:
            df,globals()['old_array_%s'%player], globals()['cdist_%s'%player] = player_localization(df,frame,player,globals()['old_array_%s'%player], globals()['cdist_%s'%player])
    return df
​
# SAVE EXAMPLE TO FOLDER:
# f = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/test_image.jpg"
​
In [124]:

#name = "#48 RHP Tom Flippin 6-3 GO"
path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/"
outputs = path# + "handle_one_old/"
argmin_list = []
# df = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one/"+name+".csv", dtype = {"Frame": np.int32, "Batter": np.ndarray})
for json_file in ["490987-79a32c9c-d4dc-4a63-aae9-804f697eb56d.json"]: #listdir(outputs):
    name = json_file[:-5] #json_file.split("_")[0]
    if json_file[-5:]!= ".json" or json_file[-10:]=="local.json": # or name+".mp4" in listdir(outputs):
        print("wrong", json_file)
        continue
    print(name+".mp4")
    with open(path+json_file, "r") as infile:
        handle_one_arr = json.load(infile)
    #handle_one_arr = from_json(outputs+ name+".json")
    if name+".mp4" in listdir(path+"batter"):
        f = path+"batter/"+name+".mp4"
    else:
        f = path+name+".mp4"
    #name = "#5 RHP Matt Blais (4)"
    print(len(handle_one_arr))
    print(np.array(handle_one_arr[0]).shape)
    print(f)

    for i in range(len(handle_one_arr)):
        handle_one_arr[i]=np.array(handle_one_arr[i])
    #with open(outputs+name+"_local.json", "w") as outfile:
     #   json.dump(local_out, outfile)

    arr = color_video(handle_one_arr, f, start = 118, end = 130, point = 2, plotting = True)#len(handle_one_arr)) #, printing = argmin_list)
    #io.vwrite("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one_bsp.mp4", arr)
    break
    df_handle = [handle_one_arr, handle_one_arr.copy()]
    # print(handle_one_arr[0])
    # CENTER WITH BOTTOM_TOP; LEFT_RIGHT
    center = [141,108] #[dictionary[name][1],dictionary[name][0]] #[83, 102] # [ 182.,  601.] #
    new_df = np.array(df_coordinates(df_handle, np.array([center]), player_list)[1])
    print("out_df", np.array(new_df).shape)
​
    # arr2 = color_video(new_df[:,:12,:], f, start = 130, end =135, point = 2, printing = None, plotting=True)
    #io.vwrite(outputs+name+"_kalmann_stronger.mp4", arr2)



#name = "#48 RHP Tom Flippin 6-3 GO"
path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/"
outputs = "./sv/v0testing/" # + "handle_one_old/"
argmin_list = []
# df = pd.read_csv("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one/"+name+".csv", dtype = {"Frame": np.int32, "Batter": np.ndarray})
for json_file in ["#21 Isaac Feldstein Home run swing.json"]: #listdir(outputs):
    name = json_file[:-5]
    if json_file[-5:]!= ".json" or json_file[-10:]=="local.json": # or name+".mp4" in listdir(outputs):
        print("wrong", json_file)
        continue
    print(name+".mp4")
    #with open(outputs+ name+".json", "r") as infile:
     #   handle_one_arr = json.load(infile)
    handle_one_arr = from_json(outputs+ name+".json")
    if name+".mp4" in listdir(path+"batter"):
        f = path+"batter/"+name+".mp4"
    else:
        f = path+"pitcher/"+name+".mp4"
    #name = "#5 RHP Matt Blais (4)"
    print(handle_one_arr.shape)
    print(np.array(handle_one_arr[0]).shape)
    print(f)

    #local_out = [elem.tolist() for elem in new_df]

    #with open(outputs+name+"_local.json", "w") as outfile:
     #   json.dump(local_out, outfile)


    #print(handle_one_arr[200:205])
    arr = color_video(handle_one_arr, f, start = 290, end = 300, cut_frame = False, plotting = True)#len(handle_one_arr)) #, printing = argmin_list)
    break
    # skvideo.io.vwrite("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/handle_one_bsp.mp4", arr)
    for i in range(12):
        for j in range(2):
            new_df[:,i,j] = kalmann(new_df[:,i,j]) #, window_len = 12, window = "flat")

    df_handle = [handle_one_arr, handle_one_arr.copy()]

    center = [dictionary[name][1],dictionary[name][0]] #[83, 102] # [ 182.,  601.] #
    new_df = np.array(df_coordinates(df_handle, np.array([center]), player_list)[1])
    print("out_df", np.array(new_df).shape)
    print(f)
    # new_df = ndimage.filters.gaussian_filter1d(np.array(new_df), axis = 0, sigma = 3)
    print(new_df.shape)
    plt.figure(figsize=(20,10))
    plt.plot(new_df[:,:12,0])
    plt.title("estelle")
    plt.show()
    arr2 = color_video(new_df[:,:12,:], f, start = 0, end =len(new_df), printing = None, plotting=False)
    io.vwrite(outputs+name+"_kalmann_stronger.mp4", arr2)



def color_video_two(json_array1, json_array2, vid_file, start = 0, end = 300, printing =None, plotting=True):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 170, 255],
              [0, 0, 0],
          [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255],
              [0, 85, 255],  [85, 0, 255],
          [170, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [0, 255, 170],  [0, 170, 255], [0, 85, 255],  [85, 0, 255], \
          [170, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors_string = ["blue", "green", "red", "tuerkis", "pink", "yellow", "orange", "black", "purple"]
    nr_joints =12
    #print(json_array.shape)
    #writer = cv2.VideoWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi",cv2.VideoWriter_fourcc(*"XVID") , 20, (500,800))

    #writer = io.FFmpegWriter("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/test.avi", (10,800,500,3))
    #writer.open()
    video_capture = cv2.VideoCapture(vid_file)
    print(vid_file)
    if start!=0:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    arr = [] #np.zeros((100,800,500,3))

    bottom_b = 265
    left_b =499
    right_b= 703
    top_b = 99

    # fig = plt.figure(figsize=(5, 15)) # for subplots
    for k in range(start, end):
        #print(k)
        if printing!=None:
            #print("dist_min",  "ratio_min")
            #print(colors_string[printing[k][0]], colors_string[printing[k][1]])
            print(printing[k])
        ret, frame = video_capture.read()
        if frame is None:
            print("end", k)
            break
        if len(json_array1[k].shape)==2:
            all_peaks = np.reshape(json_array1[k], (12, 1,2))
            all_peaks2 = np.reshape(json_array2[k], (12, 1,2))
        else:
            all_peaks = json_array1[k]
            all_peaks2 = json_array2[k]
        #print(all_peaks.shape)

        canvas = frame #[top_b:bottom_b, left_b:right_b] # cv2.imread(f) # B,G,R order
        oriImg = canvas.copy()

        for i in range(len(all_peaks)):
            #print("person", all_peaks[i])
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, (int(all_peaks[i,j,0]),int(all_peaks[i,j,1])) , 2, colors[i], thickness=-1)
                cv2.circle(canvas, (int(all_peaks2[i,j,0]),int(all_peaks2[i,j,1])) , 2, colors[i], thickness=-1)


        to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
        arr.append(to_plot[:,:,[2,1,0]])
        if plotting:
            plt.imshow(to_plot[:,:,[2,1,0]])
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(12, 12)
            plt.show()
    return np.array(arr)


def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    with open(file, 'r') as inf:
        out = json.load(inf)

    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)

    return np.array(liste)

game = "fb2d39a6-49f9-4204-969b-1e0fbdfab7da.mp4"
folder = "sv/test_outputs/"
#print(folder+game[:-4]+"_batter.json")
#print(game[:-4]+"_batter.json" in listdir("sv/"))
print(listdir("sv/test_outputs/"))
new_df = from_json(folder+game[:-4]+"_batter.json")

#print(new_df.shape)
print(new_df[:20])
f = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/"+game #83764a69-2028-4530-8188-f7c37162d403.mp4"

new_df2 = from_json(folder + game[:-4]+"_pitcher.json")
print(new_df.shape, new_df2.shape)
arr2 = color_video_two(new_df[:,:12,:],new_df2[:,:12,:], f, start = 0, end =len(new_df), printing = None, plotting=False)
print(arr2.shape)
#io.vwrite(folder+game, arr2)



folder = "/Volumes/Nina Backup/outputs/old_videos/sv/"
dic = {"490795": "2017-05-24", "490493":"2017-05-02", "490987":"2017-06-07", "491122":"2017-06-17", "491001":"2017-06-08", "491465":"2017-07-16"}
if folder[-3:]=="cf/":
    ext = ".mp4"
    view = "center field/"
else:
    ext = ".m4v"
    view = "side view/"
folder_list = listdir(folder)
for game in folder_list:
    if "batter" in game:
        print(game)
        continue

    print(dic[game[:6]])
    video_folder = "/Volumes/Nina Backup/videos/atl/"+dic[game[:6]]+"/"+view
    #print(listdir(video_folder))
    name = game.split("_")[0]
    if name+ext in folder_list:
        print("already there", name)
        continue
    new_df = from_json(folder+name+"_batter.json")
    new_df2 = from_json(folder + name+"_pitcher.json")

    f = video_folder+name+ext #83764a69-2028-4530-8188-f7c37162d403.mp4"

    print(new_df.shape, new_df2.shape)
    arr2 = color_video_two(new_df[:,:12,:],new_df2[:,:12,:], f, start = 0, end =len(new_df), printing = None, plotting=False)
    print(arr2.shape)
    io.vwrite(folder+name+ext, arr2)


​
in_folder = listdir("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher")
for i in in_folder:
    if i[:-4] not in dictionary.keys():
        print(i)
In [ ]:

​
In [ ]:

path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/batter/"
f = '#25 Trevor DeMerritt rbi base hit.mp4'
print(f[:-4] in dictionary.keys())
​
center = dictionary[f[:-4]] #[1200,650]
cap  = cv2.VideoCapture(path+f)
ret, frame = cap.read()
cv2.circle(frame, (int(center[0]),int(center[1])) , 15, [255, 0, 0], thickness=-1)
plt.imshow(frame)
plt.show()
​
dictionary[f[:-4]]=center
Test for all files
In [ ]:

path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
​
for fi in dictionary.keys():
    f = fi+'.mp4'
    print(f)
    # print(f[:-4] in dictionary.keys())
    if f in listdir("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/batter"):
        continue
    center = dictionary[fi] #[700,650]
    print(center)
​
    cap  = cv2.VideoCapture(path+f)
    ret, frame = cap.read()
    cv2.circle(frame, (int(center[0]),int(center[1])) , 15, [255, 0, 0], thickness=-1)
    plt.imshow(frame)
    plt.show()
​
    #dictionary[f[:-4]]=center
In [ ]:

with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/center_dics.json", "w") as outfile:
    json.dump(dictionary, outfile)
In [13]:

with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/center_dics.json", "r") as infile:
    dictionary = json.load(infile)
In [ ]:

​
In [ ]:

path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
f = '#5 RHP Matt Blais (3).mp4'
print(f[:-4] in dictionary.keys())
​
center = dictionary[f[:-4]] #[1200,650]
cap  = cv2.VideoCapture(path+f)
i = 0
while True:
    ret, frame = cap.read()
    if frame is None:
        print(i-1)
        plt.imshow(prev_frame)
        plt.show()
        break
    # cv2.circle(frame, (int(center[0]),int(center[1])) , 15, [255, 0, 0], thickness=-1)
    prev_frame = frame
    i+=1
In [ ]:

with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/test_functions.json","r") as infile:
    images = json.load(infile)
print(len(images))
In [ ]:

for i in images:
    plt.imshow(i)



# RELEASE Frame


import cv2
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import json
from scipy.signal import argrelextrema
from scipy import ndimage
import ast
In [2]:

from data_preprocess import Preprocessor
prepro = Preprocessor("cf_data.csv")
prepro.remove_small_classes(10)
joints_array = prepro.get_coord_arr()
print(joints_array.shape)
​
release = prepro.get_release_frame(60,120)
assert(len(release)==len(joints_array))
files_csv = prepro.cf["play_id"].values
assert(len(files_csv)==len(joints_array))
/Users/ninawiedemann/anaconda/lib/python3.5/site-packages/IPython/core/interactiveshell.py:2821: DtypeWarning: Columns (253,254,255,256,257,258,259,289) have mixed types. Specify dtype option on import or set low_memory=False.
  if self.run_code(code, result):


just_max = []
with_grad = []
with_surrounding_grad = []
higher_shoulders = []
​
surround_range = 5
joints_list = ["right_shoulder", "left_shoulder", "right_elbow", "right_wrist","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck ",
            "right_eye", "right_ear","left_eye", "left_ear"]
smooth_all = ndimage.filters.gaussian_filter1d(joints_array, axis=1, sigma = 3)
grad_all = np.gradient(smooth_all, axis=1)
​
nr = 50
for j in range(nr): #len(joints_array)):
    print(files_csv[j])
    print("truth:", release[j])
    right_wrist = joints_array[j,:,3,0]
​
    # just_max
    just_max.append(np.argmin(right_wrist))
    highest_ind = np.array(argrelextrema(right_wrist, np.less))[0] # np.argsort(right_wrist)
    highest = highest_ind[np.argsort(right_wrist[highest_ind])]
    print("min", np.argmin(right_wrist), "argsort", np.argsort(right_wrist)[:5], "argextrema", highest[:5])
​
    # with_grad:
    smooth = ndimage.filters.gaussian_filter1d(right_wrist, sigma = 3)
    grad = np.gradient(smooth)
    grad2 = np.gradient(np.gradient(smooth))
    # maxima = argrelextrema(grad, np.greater)
    # minima = argrelextrema(grad, np.less)
    extrema = np.where((grad2[:-1] * grad2[1:]) < 0)[0]
    #print(extrema)
    for i in range(5):
        if np.sum(np.absolute(highest[i]-extrema)<10)>1:
            #print("found")
            #print(highest[-i])
            with_grad.append(highest[i])
            #print("differences", np.absolute(highest[-i]-extrema))
            #print("smaller5:", np.sum(np.absolute(highest[-i]-extrema)<10))
            break
        else:
            #print("not found")
            if i==4:
                with_grad.append(0)
        #print(highest[-i])
        #print("differences", np.absolute(highest[-i]-extrema))
        #print("smaller5:", np.sum(np.absolute(highest[-i]-extrema)<10))
​
    # surrounding_grad:
    highest_5 = list(highest_ind) #[-4:])
    highest_5.append(np.argmin(right_wrist))
    # print("highest_indizes", highest_5)
    surrounding_grad = []
    for i in range(len(highest_5)):
        if highest_5[i]<5 or highest_5[i]>160:
            surrounding_grad.append(0)
        else:
            up_down_grad = - np.mean(grad[highest_5[i]-surround_range:highest_5[i]])+ np.mean(grad[highest_5[i]:highest_5[i]+surround_range])
            surr_five_grad = np.mean(np.absolute(grad[highest_5[i]-5:highest_5[i]+5]))
            surrounding_grad.append(up_down_grad)
    print("surrounding_grad", highest_5[np.argmax(surrounding_grad)])
    with_surrounding_grad.append(highest_5[np.argmax(surrounding_grad)])
​
    # higher shoulders:
    wrist_ellbow_right = np.mean(joints_array[j, :, 2:4, 0], axis = 1) # y coordinate of ellbow and wrist
    wrist_ellbow_left = np.mean(joints_array[j, :, 4:6, 0], axis = 1)
    shoulders = np.mean(joints_array[j,: ,:2,0], axis = 1) # y coordinate of shoulders
    if min(wrist_ellbow_right-shoulders)<min(wrist_ellbow_left-shoulders):
        higher = np.argmin(wrist_ellbow_right-shoulders)
        print("higher shoulders right", higher)
    else:
        higher = np.argmin(wrist_ellbow_left-shoulders)
        print("higher shoulders left", higher)
​
    higher_shoulders.append(higher)
​
​
​
    if abs(release[j]-higher)>3:
        # Plotting
        fig = plt.figure(figsize = (20,10))
        #for i in range(12):
         #   plt.plot(joints_array[j,:,i,0], label = joints_list[i]) #, joints_array[i,:,3,0])
        plt.plot(right_wrist)
        #plt.plot(np.gradient(joints_array[j,:,3,1]))
        plt.legend()
        plt.show()
​
    frame_nr = higher
    f = "/Volumes/Nina Backup/videos/atl/2017-06-07/center field/490987-"+files_csv[j]+".mp4"
    path_input_dat=f+'.dat'
    for i in open(path_input_dat).readlines():
        datContent=ast.literal_eval(i)
    bottom_p=datContent['Pitcher']['bottom']
    left_p=datContent['Pitcher']['left']
    right_p=datContent['Pitcher']['right']
    top_p=datContent['Pitcher']['top']
    cap = cv2.VideoCapture(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ret, frame = cap.read()
    #print(joints_arr[i][3][1])
    frame = frame[top_p:bottom_p, left_p:right_p]
    for l in range(12):
        cv2.circle(frame, (int(joints_array[j,frame_nr,l,1]),int(joints_array[j,frame_nr,l,0])) , 2, (250,0,0), thickness=-1)
    plt.imshow(frame)
    plt.show()


In [73]:

dic = {"just_max": just_max, "with_grad": with_grad, "with_surrounding_grad":with_surrounding_grad, "higher_shoulders":higher_shoulders}
for key in list(dic.keys()):
    liste = dic[key]
    correct = np.sum(np.array(np.absolute(liste-release[:nr]))<3)
    print(key, correct/float(nr))
with_surrounding_grad 0.748
with_grad 0.47
just_max 0.58
In [130]:

def surround_grad_method(frames, joint =3):
    right_wrist = 700-np.array(frames[:,joint,0])
    highest_ind = np.array(argrelextrema(right_wrist, np.greater))[0] # np.argsort(right_wrist)
    highest = highest_ind[np.argsort(right_wrist[highest_ind])]
    #print("max", np.argmax(right_wrist), "argsort", np.argsort(right_wrist)[-5:], "argextrema", highest[-5:])
​
    # with_grad:
    smooth = ndimage.filters.gaussian_filter1d(right_wrist, sigma = 3)
    grad = np.gradient(smooth)
​
    highest_5 = list(highest_ind) #[-4:])
    highest_5.append(np.argmax(right_wrist))
    #print(highest_5)
    surrounding_grad = []
    for i in range(len(highest_5)):
        if highest_5[i]<5: # or highest_5[i]>160:
            surrounding_grad.append(0)
        else:
            up_down_grad = np.mean(grad[highest_5[i]-surround_range:highest_5[i]])- np.mean(grad[highest_5[i]:highest_5[i]+surround_range])
            surr_five_grad = np.mean(np.absolute(grad[highest_5[i]-5:highest_5[i]+5]))
            surrounding_grad.append(up_down_grad)
    #print(surrounding_grad, highest_5[np.argmax(surrounding_grad)])
    # with_surrounding_grad.append(highest_5[np.argmax(surrounding_grad)])
    #if abs(release[j]-highest_5[np.argmax(surrounding_grad)])>3:
    # Plotting
    fig = plt.figure(figsize = (20,10))
    #for i in range(12):
        #plt.plot(joints_array[j,:,i,1], label = joints_list[i]) #, joints_array[i,:,3,0])
    plt.plot(smooth)
    #plt.plot(np.gradient(joints_array[j,:,3,1]))
    #plt.legend()
    plt.show()
    return highest_5[np.argmax(surrounding_grad)]
In [131]:

def new_data(directory):
    from os import listdir
    out_arr = []
    files = []
    dire = listdir(directory)
    for filename in dire:
        # filename = fi.split("_")[0] #"#33 Logan Trowbridge"
        #print(filename)
        if filename[-4:]==".mp4" or filename[0]==".":
            continue
        if filename.split("_")[0]+".mp4" in listdir("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/batter"):
            continue
        files.append(filename.split("_")[0])
        coordinates = ["x", "y"]
        joints_list = ["right_shoulder", "left_shoulder", "right_elbow", "right_wrist","left_elbow", "left_wrist",
                "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck ",
                "right_eye", "right_ear","left_eye", "left_ear"]
​
        def from_json(file):
            with open(file, 'r') as inf:
                out = json.load(inf)
​
            liste = []
            for fr in out["frames"]:
                l_joints = []
                for j in joints_list[:12]:
                    l_coo = []
                    for xy in coordinates:
                        l_coo.append(fr[j][xy])
                    l_joints.append(l_coo)
                liste.append(l_joints)
            return np.array(liste)
​
        out_arr.append(from_json("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/"+filename))
    return out_arr, files
​
def higher_shoulders_max(joints_array):
    wrist_ellbow_right = np.mean(joints_array[:, 2:4, 0], axis = 1) # y coordinate of ellbow and wrist
    wrist_ellbow_left = np.mean(joints_array[:, 4:6, 0], axis = 1)
    shoulders = np.mean(joints_array[: ,:2,0], axis = 1) # y coordinate of shoulders
    if min(wrist_ellbow_right-shoulders)<min(wrist_ellbow_left-shoulders):
        higher = np.argmin(wrist_ellbow_right-shoulders)
        print("higher shoulders right", higher)
        return higher
    else:
        higher = np.argmin(wrist_ellbow_left-shoulders)
        print("higher shoulders left", higher)
        return higher
In [161]:

from os import listdir
directory = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/outputs_example/"
vid_dir = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
new_joints_arr, files = new_data(directory)
#print(len(new_joints_arr))
#print(files)
for i, arr in enumerate(new_joints_arr):
    #arr = ndimage.filters.gaussian_filter1d(np.array(arr), axis=0, sigma = 2)
    #print(arr)
    fig = plt.figure(figsize = (20,10))
    for k in range(12):
    #k = 3
        plt.plot(arr[:,k,0], label = joints_list[k]) #, joints_array[i,:,3,0])
    #plt.plot(np.mean(arr[:, 2:4, 0], axis = 1), label = "wrist_ellbow")
    #plt.plot(np.mean(arr[:, :2, 0], axis = 1), label = "shoulders")
    wrist_ellbow = np.mean(arr[:500, 2:4, 0], axis = 1) # y coordinate of ellbow and wrist
    shoulders = np.mean(arr[:500,:2,0], axis = 1) # y coordinate of shoulders
    plt.plot(wrist_ellbow-shoulders)
    frame_nr = higher_shoulders_max(arr) #np.argmin(wrist_ellbow-shoulders)
    #plt.ylim(900,100)
    plt.legend()
    plt.show()
    # frame_nr = np.argmin(np.array(arr)[:,3,0]) #surround_grad_method(arr)
    print(frame_nr)
    #print(files[i]+".mp4" in listdir(directory))
    cap = cv2.VideoCapture(vid_dir+files[i]+".mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ret, frame = cap.read()
    print(new_joints_arr[i][3][1])
    cv2.circle(frame, (int(new_joints_arr[i][frame_nr][3][1]),int(new_joints_arr[i][frame_nr][3][0])) , 8, (250,0,0), thickness=-1)
    plt.imshow(frame)
    plt.show()


"""path = "/Volumes/Nina Backup/videos/atl/2017-04-19/center field/"
vid_example = "490324-0ade0abd-fb64-49ca-aacd-66a50901f538.mp4"
cap = cv2.VideoCapture(path+vid_example)
​
for i in open(path+vid_example+".dat").readlines():
    datContent=ast.literal_eval(i)
​
bottom_b=datContent['Batter']['bottom']
left_b=datContent['Batter']['left']
right_b=datContent['Batter']['right']
top_b=datContent['Batter']['top']
bottom_p=datContent['Pitcher']['bottom']
left_p=datContent['Pitcher']['left']
right_p=datContent['Pitcher']['right']
top_p=datContent['Pitcher']['top']
cap.set(cv2.CAP_PROP_POS_FRAMES, 85)
images_pitcher = []
images_batter = []
for i in range(10):
    ret, frame = cap.read()
    images_pitcher.append(frame[top_p:bottom_p, left_p:right_p])
    images_batter.append(frame[top_b:bottom_b, left_b:right_b])"""
​
def color_box(frame, bbox, color = "red"):
    """
    vid: path to video or image file
    bbox: [oben, unten, links, rechts] bzw [kleineres_y, größeres_y, kleineres_x, größeres_x]
    kann sein dass das bei dir anderes ist, bei mir ist eben die y achse immer die werte umgedreht,
    also oben sind kleinere werte als unten
    """
    # for video:
    #video_capture = cv2.VideoCapture(vid)
    #ret, frame = video_capture.read()
    # for image
    #frame = plt.imread(vid)
​
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame, aspect='equal')
​
    ax.add_patch(
    plt.Rectangle((int(bbox[2]), int(bbox[0])), # linkes oberes eck
                  int(bbox[3]-bbox[2]), # länge von der seite von oben nach unten
                  int(bbox[1]-bbox[0]), fill=False, #länge von der seite von links nach rechts
                  edgecolor=color, linewidth=3.5)
    )
    plt.show()

path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"
vid_example = "#31 LHP Michael Chavez (2).mp4"
cap = cv2.VideoCapture(path+vid_example)
with open("/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/center_dics.json", "r") as infile:
    dictionary = json.load(infile)
center = dictionary[vid_example[:-4]]
print(center)
bbox = [center[1]-400, center[1]+300, center[0]-1100, center[0]-300]
cap.set(cv2.CAP_PROP_POS_FRAMES, 162)
images = []
for i in range(8):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = frame_gray[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    #b =frame_gray>200
    plt.imshow(a)
    plt.gray()
    plt.show()
    images.append(a) #frame_gray[bbox[0]:bbox[1], bbox[2]:bbox[3]])
    #color_box(frame, bbox)
#bbox = [center[0]]
[1650, 600]








In [43]:

path = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/"
vid_example = "pitcher/#31 LHP Michael Chavez (2).mp4"
​
def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    with open(file, 'r') as inf:
        out = json.load(inf)
​
    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)
​
    return np.array(liste)
​
​
​
bbox = [center[1]-400, center[1]+300, center[0]-1100, center[0]]
​
joints_array = from_json(path+"outputs_example/"+"#31 LHP Michael Chavez (2)_joints.json")
print(joints_array.shape)
print("ellbow wrist", joints_array[160, 4:6])
​
cap = cv2.VideoCapture(path+vid_example)
​
cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
​
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
​
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
​
# Create some random colors
color = np.random.randint(0,255,(100,3))
​
# Take first frame and find corners in it
ret, old_frame_in = cap.read()
old_frame = old_frame_in#[bbox[0]:bbox[1], bbox[2]:bbox[3]]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#print(p0)
#p0 = np.roll(np.array([[joints_array[160,4]], [joints_array[160,5]]]), 1, axis = 2)
#print(p0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
​
​
j=0
while j<10:
    ret,frame_in = cap.read()
    frame = frame_in#[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
​
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
​
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
​
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)


    plt.figure(figsize = (20,10))
    plt.imshow(img)
    plt.show()

​
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    j+=1
​
plt.figure(figsize = (20,10))
plt.imshow(img)
plt.show()
(264, 12, 2)
ellbow wrist [[  493.5  1440.5]
 [  468.5  1511. ]]



In [44]:

import tensorflow as tf
Tensor("Const:0", shape=(3, 2, 3), dtype=int32)
In [121]:

a = images[2] #np.zeros((200,200))
#a[50:150,50:150]=1
window_size = 100
maxi = np.inf
maxi_ind = (i,j)
maxi_img = []
var_img = []
for i in range(window_size//2, len(a)-window_size//2, 3):
    maxi_img_inner = []
    var_img_inner = []
    for j in range(window_size//2, len(a[0])-window_size//2, 3):
        # print(i,j)
        patch = a[i-window_size//2:i+window_size//2, j-window_size//2:j+window_size//2]
        blur = cv2.Laplacian(patch, cv2.CV_64F).var()
        var_img_inner.append(np.std(patch))
        maxi_img_inner.append(blur)
        if blur<maxi:
            maxi= blur
            maxi_ind = (i,j)
    maxi_img.append(maxi_img_inner)
    var_img.append(var_img_inner)
​
print(maxi_ind)
#print(maxi_img)
#print(cv2.Laplacian(a, cv2.CV_64F).var())
(83, 56)
In [122]:

#print(var_img)
var_img = np.array(var_img)
plt.imshow(var_img)
plt.show()
var_img_new = var_img/np.max(var_img)
maxi_img = np.array(maxi_img)
#maxi_img = 20-maxi_img
#maxi_img[maxi_img<0]==0
plt.imshow(maxi_img)
plt.show()
maxi_img_new = 1-(maxi_img/np.max(maxi_img))
result = (maxi_img_new+var_img_new)/2
plt.imshow(result)
plt.show()



In [108]:

from scipy import ndimage
for i, img in enumerate(images):
    img = ndimage.filters.gaussian_filter(img, sigma = 20)
    if i>0:
        diff = np.absolute(before-img)
        diff_smooth = ndimage.filters.gaussian_filter(diff, sigma = 20)
        #print(diff)
        plt.imshow(diff_smooth)
        plt.show()
    before = img





# We read the images as grayscale so they are ready for thresholding functions.
#fig = plt.figure(figsize=(18, 16), edgecolor='k')
#plt.imshow(images[0])
#plt.show()


def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    with open(file, 'r') as inf:
        out = json.load(inf)

    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)

    return np.array(liste)

# PARAMETERS
def detect_ball(folder, template = "%03d.jpg", min_area = 400, plotting=True):
    # length = len(os.listdir(folder))
    # images = [cv2.imread(folder+template %idx, cv2.IMREAD_GRAYSCALE) for idx in range(length)] #IMG_TEMPLATE.format(idx), )

    cap = cv2.VideoCapture(folder)
    images=[]
    length=0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        length+=1


    start = time.time()

    candidate_values = []
    whiteness_values = []
    location = []
    frame_indizes = []
    for t in range(1, length-1):
        im_tm1 = images[t - 1]
        im_t = images[t]
        im_tp1 = images[t + 1]
        delta_plus = cv2.absdiff(im_t, im_tm1)
        delta_0 = cv2.absdiff(im_tp1, im_tm1)
        delta_minus = cv2.absdiff(im_t,im_tp1)
        sp = cv2.meanStdDev(delta_plus)
        sm = cv2.meanStdDev(delta_minus)
        s0 = cv2.meanStdDev(delta_0)
        # print("E(d+):", sp, "\nE(d-):", sm, "\nE(d0):", s0)

        th = [
            sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
            sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
            s0[0][0, 0] + 3 * math.sqrt(s0[1][0, 0]),
        ]

        #print("Thresholds:", th)
        """
        # NAIVE THRESHOLD
        start = time.time()
        dbp = threshold(delta_plus, th[0])
        dbm = threshold(delta_minus, th[1])
        db0 = threshold(delta_0, th[2], invert=True)

        detect_naive = combine(dbp, dbm, db0)
        naive_time = (time.time() - start) * 1000
        """

        # OPENCV THRESHOLD

        ret, dbp = cv2.threshold(delta_plus, th[0], 255, cv2.THRESH_BINARY)
        ret, dbm = cv2.threshold(delta_minus, th[1], 255, cv2.THRESH_BINARY)
        ret, db0 = cv2.threshold(delta_0, th[2], 255, cv2.THRESH_BINARY)

        detect = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(dbp, dbm),
                        cv2.bitwise_not(db0)))

        # CONNECTED BOX
        # The original `detect` image was suitable for display, but it is "inverted" and not suitable
        # for component detection; we need to invert it first.
        start = time.time()
        nd = cv2.bitwise_not(detect)
        # only stats is used, not num, labels, centroids
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)
        # We set an arbitrary threshold to screen out smaller "components"
        # which may result simply from noise, or moving leaves, and other
        # elements not of interest.


        d = detect.copy()
        candidates = list()
        for stat in stats[1:]:
            area = stat[cv2.CC_STAT_AREA]
            if area < min_area:
                continue # Skip small objects (noise)

            lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
            rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
            bottomLeftCornerOfText = (lt[0], lt[1] - 15)

            candidates.append((lt, rb, area))

        #plt.figure(figsize=(10, 10), edgecolor='k')

        # MOTION MODEL

        # Thinning threshold.
        psi = 0.7

        # Area matching threshold.
        gamma = 20 #0.4 changed to see all candidates
        area_max = 0

        fom_detected = False
        start = time.time()
        col = len(candidates)
        #print("number of candidates", col)
        sub_regions = list()

        index = []
        for i, candidate in enumerate(candidates):
            # The first two elements of each `candidate` tuple are
            # the opposing corners of the bounding box.
            x1, y1 = candidate[0]
            x2, y2 = candidate[1]


            # We had placed the candidate's area in the third element of the tuple.
            actual_area = candidate[2]

            # For each candidate, estimate the "radius" using a distance transform.
            # The transform is computed on the (small) bounding rectangle.
            cand = nd[y1:y2, x1:x2]
            dt = cv2.distanceTransform(cand, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
            radius = np.amax(dt)

            # "Thinning" of pixels "close to the center" to estimate a
            # potential FOM path.
            ret, Pt = cv2.threshold(dt, psi * radius, 255, cv2.THRESH_BINARY)

            # TODO: compute actual path lenght, using best-fit straight line
            #   along the "thinned" path.
            # For now, we estimate it as the max possible lenght in the bounding box, its diagonal.
            w = x2 - x1
            h = y2 - y1
            path_len = math.sqrt(w * w + h * h)
            expected_area = radius * (2 * path_len + math.pi * radius)

            area_ratio = abs(actual_area / expected_area - 1)
            #print(area_ratio)

            location.append([(x1+x2)/2, (y1+y2)/2])
            index.append(i)
            area = candidates[i][2]
            candidate_values.append(area_ratio)
            patch = im_t[y1:y2, x1:x2]
            #plt.figure()
            #plt.imshow(patch)
            #plt.show()
            whiteness_values.append(np.mean(patch))
            frame_indizes.append(t)

        if plotting and len(candidates)>0:
            print("DETECTED", t, whiteness_values[-1], candidate_values[-1])
            plt.figure(figsize=(10, 10), edgecolor='r')
            # print(candidates[fom])
            img = np.tile(np.expand_dims(im_t, axis = 2), (1,1,3))
            print(img.shape)
            for fom in index:
                cv2.rectangle(img, candidates[fom][0], candidates[fom][1],[255,0,0], 2)
            plt.imshow(img, 'gray')
            plt.title("Detected FOM".format(t))
            plt.show()

    print("time for %s frames without plotting"%length, (time.time() - start) * 1000)
    return frame_indizes, candidate_values, whiteness_values, location

example = "#00 RHP Devin Smith"# "#26 RHP Tim Willites"
BASE = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"+example+".mp4" #"data/Matt_Blais/" # für batter: pic davor, 03 streichen und (int(idx+1))
joints_path = "/Volumes/Nina Backup/high_quality_outputs/"+example+".json"

BASE = "/Volumes/Nina Backup/CENTERFIELD/4f7477b1-d129-4ff7-a83f-ad322de63b24.mp4"
joints_path = "/Volumes/Nina Backup/outputs/new_videos/cf/490770_4f7477b1-d129-4ff7-a83f-ad322de63b24_pitcher.json"
joints = from_json(joints_path)[:,:12,:]
print(joints.shape)
frame_indizes, candidate_values, whiteness_values, location = detect_ball(BASE, min_area=10)



#print(candidate_values)
#print(np.argmax(candidate_values))
#print(np.argsort(candidate_values))
#print(frame_indizes)
#print(whiteness_values)
#print(delta_plus.shape, delta_0.shape, delta_minus.shape)
location = np.array(location)

def find_consecutive_frame():
    count =0
    frame = 0
    results=[]
    for detection in range(len(frame_indizes)):
        same_frame = frame_indizes[detection]==frame
        frame = frame_indizes[detection]
        bbox = [np.min(joints[frame, :, 0]), np.max(joints[frame,:, 0]),
                       np.min(joints[frame, :, 1]), np.max(joints[frame, :, 1])]
        if bbox[0]<location[detection, 0]<bbox[1] and bbox[2]<location[detection, 1]<bbox[3]:
            if not same_frame:
                count = 0
            continue
        #distance = min([np.linalg.norm(location[detection]-joints[frame, i]) for i in range(12)])
        if np.all(joints[frame, :, 1]>location[detection,1]):
            count+=1
        elif not same_frame:
            count = 0

        if count==3:
            print("found three points after another", frame)
            results.append(frame)
    return results

res = find_consecutive_frame()
print("results of find_consecutive_frame", res)

# find highest detections
a = np.argsort(location[:,1])
x = np.array(frame_indizes)[a]
result = np.median(x[:5])
print("results of median of lowest frames", result)

# find consecutive frames in highest ones
sort_x = np.sort(x[:10])
print(sort_x)
count=0
for i in range(1, len(sort_x)):
    if abs(sort_x[i]-sort_x[i-1])==1:
        count+=1
    else:
        count=0
    if count>2:
        print("found consecutive frame", sort_x[i-2])

print("just argmin", frame_indizes[np.argmin(location[:,1])])



plt.scatter(frame_indizes, candidate_values)
plt.show()
plt.scatter(frame_indizes, whiteness_values)
plt.show()
plt.scatter(frame_indizes, location[:,1])
plt.plot(joints[:, 2,1], color = "green")
plt.show()
plt.scatter(frame_indizes, location[:,0])
plt.plot(joints[:, 2,0], color = "green")
plt.show()
