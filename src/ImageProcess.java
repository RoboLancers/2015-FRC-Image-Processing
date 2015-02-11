import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageProcess{
	
	   public static void main( String[] args )
	   {
	      System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
	     
	      //testConvertToGreyscale();
	      
	      writeToFile(testHoughTransform(getInputFromFile("TestImg.jpg")));
	   }
	   
	   public static void testConvertToGreyscale(){
		         System.loadLibrary( Core.NATIVE_LIBRARY_NAME );

		         Mat mat = getInputFromFile("TestImg.jpg");
		         
		         Mat mat1 = new Mat(mat.height(),mat.width(),CvType.CV_8UC1);
		         Imgproc.cvtColor(mat, mat1, Imgproc.COLOR_RGB2GRAY);

		         writeToFile(mat1);
	   }

	   public static Mat testHoughTransform(Mat img){
		   ArrayList<Double> intersectionsX = new ArrayList<Double>();
		   ArrayList<Double> intersectionsY = new ArrayList<Double>();
		   
		   
		   //Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);	
		   //Imgproc.blur(img, img, new Size(3,3));
		   
		   // Convert image to HSV space for thresholding
		   Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2HSV);
		   
		   // Old thresholding function - DOESN'T WORK ON COLOR IMAGES
		   //Imgproc.threshold(img, img, 127 , 130 , Imgproc.THRESH_BINARY);
		   
		   int low = 10;
		   int high = 255;
		   // Actual thresholding function, inputs are image in, min(H,S,V), max(H,S,V), image out
		   // Because we are looking for a silver/gray target, all values are very low
		   // For values, refer to http://en.wikipedia.org/wiki/HSL_and_HSV
		   // If we use green LEDs, will need to retune this. Change hue range to green, saturation and value high
		   Core.inRange(img, new Scalar(0, 0, 0), new Scalar(10, 10, 10), img);
		   
		   // Remove small chunks such as shadow of the box
		   Imgproc.erode(img, img, new Mat());
		   
		   //Imgproc.blur(img, img, new Size(3,3));
		   
		   //Imgproc.Canny(img, img, 100, 100);
		   
		   // Find edges - low threshold and high threshold (high threshold should actually be around 3*low threshold)
		   // FIX THIS
		   Imgproc.Canny(img, img, 1, 60);
		   
//		   Mat corners = new Mat();
//		   // Use this for corner detection - but too many stray corners to get rid of
//		   Imgproc.cornerHarris(img,corners,2,3,0.04,1);
//		   
//		   Core.normalize(corners, corners, 0, 255, Core.NORM_MINMAX,CvType.CV_32FC1, new Mat());
//		   Core.convertScaleAbs(corners, corners);
//		   
//		   int thresh = 75;
//		   
//		   double[] pixelIntensity = new double[corners.depth()];
//		   
//		   for(int i = 0; i<corners.rows(); i++){
//			   for (int j = 0; j < corners.cols(); j++){
//				   pixelIntensity = corners.get(i, j) ; 
//				   if ((pixelIntensity[0]) > thresh){
//					   Core.circle(img, new Point(j, i) ,9, new Scalar(255,0,0));
//				   }
//			   }
//		   }

		   // Detect lines in edge map, should get the edges of the L's quite well		   
		   Mat lines = new Mat();
//		   if (true){
//			   return img;
//		   }
		   // Inputs are input image, output lines, min length resolution, minimum angle resolution (currently 1 degree), 
		   // threshold, min line length, min line gap before it's two lines
		   Imgproc.HoughLinesP(img ,lines, 1, Math.PI/180, 10, 1,10);
		   
		   System.out.println("" + lines.get(0,0)[0] + "" + lines.get(0,0)[1] + "" + lines.get(0,0)[2] +"" + lines.get(0,0)[3]);
		   
		   Mat linesOut = new Mat(img.height(),img.width(),CvType.CV_8UC1);
		   
		   
		   
		   for(int i = 0; i < lines.cols();  i++){
			   double[] vec1 = lines.get(0, i);
			   double[] vecA = new double[4];
			   
			   // Draw lines on image to visualize	   
			   Core.line(linesOut, new Point(lines.get(0,i)[0], lines.get(0,i)[1]),
					   new Point(lines.get(0,i)[2], lines.get(0,i)[3]),
					   new Scalar(255,0,0,20));
			   
			   
			   vecA[0] = 0;
		       vecA[1] = (vec1[1] - vec1[3]) / (vec1[0] - vec1[2]) * -vec1[0] + vec1[1];
		       vecA[2] = img.cols();
		       vecA[3] = (vec1[1] - vec1[3]) / (vec1[0] - vec1[2]) * (img.cols() - vec1[2]) + vec1[3];

					   
			   for(int j = i + 1; j < lines.cols(); j++){
				   double[] vec2 = lines.get(0, j);
				   double[] vecB = new double[4];
				   
				   vecB[0] = 0;
			       vecB[1] = (vec2[1] - vec2[3]) / (vec2[0] - vec2[2]) * -vec2[0] + vec2[1];
			       vecB[2] = img.cols();
			       vecB[3] = (vec2[1] - vec2[3]) / (vec2[0] - vec2[2]) * (img.cols() - vec2[2]) + vec2[3];
			       
			       Point point = computeIntersection(vecA, vecB);
			       
			       if(!(point.x<0 || point.x>img.cols() || point.y<0 || point.y>img.rows()) ){
			    	   intersectionsX.add(point.x);
			    	   intersectionsY.add(point.y);
			    	   
			       }
			       
			       System.out.println("i=" + i +"j=" + j);
			       
			       
			   }
		   }
		   
		   for(int i = 0;i<intersectionsX.size();i++){
			   Core.circle(img, new Point(intersectionsX.get(i),intersectionsY.get(i)),9, new Scalar(255,0,0));
		   }
		   
		   return linesOut;
		   

	   }
	   
	   public static Mat getInputFromFile(String path){
		   File input = new File(path);
	         BufferedImage image;
			
	         try {
				image = ImageIO.read(input);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return null;
			}	

	         byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	         Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
	         mat.put(0, 0, data);
		   
	         return mat;
	   }
	   
	   public static void writeToFile(Mat mat){
		   
		   byte[] data1 = new byte[mat.rows() * mat.cols() * (int)(mat.elemSize())];
	         mat.get(0, 0, data1);
	         BufferedImage image1 = new BufferedImage(mat.cols(),mat.rows(), BufferedImage.TYPE_BYTE_GRAY);
	         image1.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data1);

	         File ouptut = new File("grayscale.jpg");
	         try {
				ImageIO.write(image1, "jpg", ouptut);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	   }
	   
	   public static Point computeIntersection (double[] vecA, double[] vecB ){
		   double x1 = vecA[0], y1 = vecA[1], x2 = vecA[2], y2 = vecA[3];
		   double x3 = vecB[0], y3 = vecB[1], x4 = vecB[2], y4 = vecB[3];
		   
		   double d = ((x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4));
		   
		   if(d != 0){
		        Point pt = new Point();
		        
		        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
		        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
		        return pt;
		    }else{
		        return new Point(-1, -1);
		    }
	   }
}
