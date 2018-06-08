all:
	pandoc meta.yaml README.md 1_Pose_Estimation/README.md 2_Movement_classification/README.md 3_Event_detection/README.md 4_Object_tracking/README.md \
	  --number-sections \
		--toc \
		-F pandoc-crossref \
		-F pandoc-citeproc \
		--bibliography library.bib \
		-o Documentation.pdf
