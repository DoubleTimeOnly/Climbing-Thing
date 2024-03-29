# Research
## Existing Projects
### Active Projects (Last Update < 6 Months Ago)
* ### [climbnet](https://github.com/juangallostra/climbnet)
    * climbnet is a CNN for hold detection and segmentation from 2D images.
    * quote from the climbnet readme: 
        > This project uses Facebook's [detectron2](https://github.com/facebookresearch/detectron2) implmentation of [Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml) and was trained using 210 images.
* ### [Interactive Rock Climbing Game](https://github.com/HarryHamilton/Computer-Vision-Rock-Climbing)
    * Intended to be a warmup game. The program takes a target and analyzes a live video feed. When it determines a climber has reached the target, they win.
    * Example: https://user-images.githubusercontent.com/8810902/145719288-dd50d741-536e-4e0a-a9d6-eca7ccae54cb.mp4
    * The program uses MediaPipe pose estimation to detect the climber, and opencv for image/video manipluation.
### Inactive Projects (Last Update > 6 Months Ago)
* ### [NeuralClimb](https://github.com/scsukas8/NeuralClimb/)
    * NeuralClimb is a project for hold detection and route grouping based on color from 2D images.
    * Project also explored analyzing video, and    
    * Uses opencv's Canny edge detector and SimpleBlobDetector
    * Cool intro/exploratory document: https://github.com/scsukas8/NeuralClimb/blob/master/Current%20Progress.pdf

## Future Ideas (Next PP Session)
* Human keypoint detection from climbing video
* Different image formats (RGB, HSV, etc.), and compare
* Different data representation (not histograms)
* Automate evaluation of clustering algorithms using different metrics
    * could use basic ranking or actual clustering


## Ideas for Route Segmentation
Given all holds on a wall, assign each hold to their respective route
* #### Average Hue Similarity
    * Not that robust
    * After testing this method can't differentiate between white, black, and the other colors, probably because those are hue independent.
        * After more testing, seems like it can't differentiate between any colors. (it's late though, so idk...)
        * Did not vary the hue threshold in testing
        
* #### Distance Metrics to try:
    * KL divergence (apparently not a true metric)
    * Jensen-Shannon metric
    * Earth Mover Distance

    All of these were not much better than cosine similarity, and EMD might have been even worse.
* #### Histogram Bin Count
     Varying bin count (180 vs 18) showed no large difference in ranking ability for the various distance metrics. For example with 180 bins cosine similiarity did better at ranking purple, but JSM was better with 18 bins. 
* #### Clustering 
    * on hold HSV histograms
        * ##### K-means with defined K
            * fewer bins (~10) seems more robust (maybe?)
            * bin count seems finiky (We do not understand its effect)
        * ##### K-means with searching for K
            * 
        * ##### DBSCAN
            * Always gave one cluster independent of epsilon.
            * learn to use better... ("git gud" -Victor)
    * on hold HSV averages
        * ##### K-means with defined K
            * unable to segmnet routes in any meaningful way. effectively randomly selects holds
            * Could be fun route generator  
                * or just use random selector?
        * ##### K-means with searching for K
            * 
        * ##### DBSCAN
            * 
* #### Hammer time / Atom Bomb
    * #### Weakly supervised clustering network
    * #### Metric learning?
    * Support Vector Machine (SVM)
    * #### Mennard (aka CLIP)
        * Try removing background. This actually sucks a lot. Does great with background, mixes up a ton of obvious colors with masking background.
        * after masking try adding our own background (e.g. a striped background -> prompt becomes "a x color object on a striped background")
