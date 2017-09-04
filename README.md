# ImageTextRecognition
Text Detection and Recognition in Natural Images

## Pipeline
### Conversion from color to gray scale
<pre><code>img = cv2.imread('test2.jpg',0) #0 indicates read in gray scale</code></pre>
### Apply MSER algorithm to get regions that probably contain text
Use MSER to extract regions that may contain text.
For more information refer to, Wikipedia: [(MSER) MAXIMALLY STABLE EXTREMAL REGIONS](https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions "Wikipedia MSER")
<pre><code>mser = cv2.MSER()
regions = mser.detect(img, None)    #detect and extract MSER lasso-contours
</code></pre>
### Form convex-hulls from the lasso contours
Convex-hulls are formed from the lasso contours
<pre><code>hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  #take convex-hull of each contour
</code></pre>
### Convert convex-hulls to bounding rectangles
The convex hulls are converted to bounded rectangles and redundant rectangles that are contained in other rectangles are removed.
<pre><code>x,y,w,h = cv2.boundingRect(hull)    #convert the hull to bounding rectangle</code></pre>
### Perform OTSU Binarization
OTSU Binarization of region under rectangle is performed to make the text stand out from the background
For more information refer to, Wikipedia: [ OTSU Binarization ](https://en.wikipedia.org/wiki/Otsu%27s_method)
<pre><code>thresh,letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)</code></pre>
For regions getting binarized into black backgrounds and white text, inversion is performed using simple heuristic
### Normalize the rectangle to standard size 64x64
<pre><code>Xnorm = [mynormalization(image,(64,64)) for image in X]</code></pre>
