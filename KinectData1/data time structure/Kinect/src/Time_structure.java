import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Calendar;


public class Time_structure {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			BufferedReader br1 = new BufferedReader(new FileReader(".\\skeleton_subject0_trial0_aditi1.csv"));
			BufferedReader br2 = new BufferedReader(new FileReader(".\\skeleton_subject0_trial0_aditi1.csv"));
			try {
				File file = new File(".\\Time_struct_subject0_trial0_aditi1.csv");
				FileWriter fw = new FileWriter(file);
				BufferedWriter bw = new BufferedWriter(fw);
				bw.write("\nDelta Timestamp,Old Label,New label,"
						+ "head.x,head.y,head.z,"
						+ "leftShoulder.x,leftShoulder.y,leftShoulder.z,"
						+ "leftElbow.x,leftElbow.y,leftElbow.z,"
						+ "leftWrist.x,leftWrist.y,leftWrist.z,"
						+ "leftHand.x,leftHand.y,leftHand.z,"
						+ "leftFingerTip.x,leftFingerTip.y,leftFingerTip.z,"
						+ "leftHip.x,leftHip.y,leftHip.z,"
						+ "leftKnee.x,leftKnee.y,leftKnee.z,"
						+ "leftAnkle.x,leftAnkle.y,leftAnkle.z,"
						+ "leftFoot.x,leftFoot.y,leftFoot.z,"
						+ "rightShoulder.x,rightShoulder.y,rightShoulder.z,"
						+ "rightElbow.x,rightElbow.y,rightElbow.z,"
						+ "rightWrist.x,rightWrist.y,rightWrist.z,"
						+ "rightHand.x,rightHand.y,rightHand.z,"
						+ "rightFingerTip.x,rightFingerTip.y,rightFingerTip.z,"
						+ "	rightHip.x,rightHip.y,rightHip.z,"
						+ "rightKnee.x,rightKnee.y,rightKnee.z,"
						+ "rightAnkle.x,rightAnkle.y,rightAnkle.z,"
						+ "rightFoot.x,rightFoot.y,rightFoot.z,"
						+ "spine.x,spine.y,spine.z,"
						+ "shoulderCenter.x,shoulderCenter.y,shoulderCenter.z,"
						+ "leftThumb.x,leftThumb.y,leftThumb.z,"
						+ "rightThumb.x,rightThumb.y,rightThumb.z,"
						+ "ballX,ballY");
				String next = new String();
				String afternframe = new String();
				next=br1.readLine();
				afternframe= br2.readLine();
				for(int i=0;i<4;i++)			//setting n= 5 frames 2 seemed too short to notice changes
				{
					afternframe=br2.readLine();
				}
				while((afternframe=br2.readLine())!=null)
				{
					next=br1.readLine();
					String part_next[]= next.split(",");
					String part_nframe[]= afternframe.split(",");
					long diff = Long.parseLong(part_nframe[0])-Long.parseLong(part_next[0]);
					bw.write("\n"+diff+","+part_next[1]+","+part_nframe[1]);
			
					for(int col=2;col<=72;col++)
					{
						System.out.println(part_nframe[col]);
						double diff_move= Double.parseDouble(part_nframe[col])-Double.parseDouble(part_next[col]);
						bw.write(","+diff_move);
					}
					
					
				}
				bw.close();
				br1.close();
				br2.close();
			}
			catch (IOException e) {
				e.printStackTrace();
			}	}
		 catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}