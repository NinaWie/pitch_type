import time
from config_reader import config_reader
param_, model_ = config_reader()

TIME_PRINT = param_['print_tictoc'] is '1'

TIME_PROBE_ID = 0
TIME_STACK = []
TEXT_PADDING = 24
FIRST_STAMP = None
PREV_STAMP = None

class TimeStamp:
    def __init__(self, label):
        self.children = []
        self.elapsed = -1
        self.begun = time.time()
        self.label = label

    def pretty(self, level=0, percentage=100.0):
        tabbing = ''.join(level * ['   '])
        equal_padding = ''.join((TEXT_PADDING - len(self.label)) * [' '])
        result = '| %s|__%s%s: %.2f%% (%.6fs)' % (tabbing, self.label, equal_padding, percentage, self.elapsed)
        return result


def time_printout(stamp, level=0):
    accounted_percentage = 0.0
    for child in stamp.children:
        elapsed_percentage = child.elapsed / stamp.elapsed* 100
        if TIME_PRINT: print child.pretty(level, elapsed_percentage)
        time_printout(child, level + 1)
        accounted_percentage += elapsed_percentage
    # TODO: percentage unaccounted for
    if len(stamp.children):
        tabbing = ''.join(level * ['   '])
        if TIME_PRINT: print '| %s---(%.2f%% unaccounted)' % (tabbing, 100 - accounted_percentage)

def time_summary():
    global FIRST_STAMP, PREV_STAMP, TIME_STACK

    if TIME_PRINT: print '| TICTOC RESULTS:'
    if TIME_PRINT: print '|'

    if TIME_PRINT: print FIRST_STAMP.pretty()
    time_printout(FIRST_STAMP, 1)

    FIRST_STAMP = None
    PREV_STAMP = None
    TIME_STACK = []

def tic(label):
    global FIRST_STAMP, PREV_STAMP

    stamp = TimeStamp(label)

    if FIRST_STAMP is None:
        FIRST_STAMP = stamp
        PREV_STAMP = stamp
    else:
        # Stamp becomes child of immediate parent; child takes over current parent
        PREV_STAMP.children.append(stamp)
        PREV_STAMP = stamp

    TIME_STACK.append(stamp)

def toc(label):
    global PREV_STAMP, TIME_STACK

    last_label = TIME_STACK[len(TIME_STACK) - 1].label
    if last_label != label:
        raise Exception('Inconsistent tic tocs: "%s" -> "%s"' % (label, last_label))

    stamp = TIME_STACK.pop()
    stamp.elapsed = time.time() - stamp.begun

    # Relinquish current parent from its role; its parent becomes the new parent
    if len(TIME_STACK):
        PREV_STAMP = TIME_STACK[len(TIME_STACK) - 1]
