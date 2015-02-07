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
		   
		   Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2HSV);
		   /*
		   
		   Imgproc.threshold(img, img, 127 , 130 , Imgproc.THRESH_BINARY);
		   
		   //Imgproc.Canny(img, img, 100, 100);
		   
		   Imgproc.Canny(img, img, 1, 60);
		   
		   Mat lines = new Mat();
		   
		   Imgproc.HoughLinesP(img ,lines, 1, Math.PI/180, 70, 50, 10);
		   
		   for(int i = 0; i < lines.cols();  i++){
			   double[] vec1 = lines.get(0, i);
			   double[] vecA = new double[4];
			   
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
		   /*
		   for(int i = 0;i<intersectionsX.size();i++){
			   Core.circle(img, new Point(intersectionsX.get(i),intersectionsY.get(i)),9, new Scalar(255,0,0));
		   }
		   */
		   return img;
		   

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
