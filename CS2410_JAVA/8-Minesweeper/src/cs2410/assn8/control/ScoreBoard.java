package cs2410.assn8.control;

import javafx.scene.control.TextInputDialog;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import static cs2410.assn8.control.ScoreBoard.Type.EASY;
import static cs2410.assn8.control.ScoreBoard.Type.HARD;
import static cs2410.assn8.control.ScoreBoard.Type.MEDIUM;

/**
 * Created by philip_nelson on 4/28/17.
 */
public class ScoreBoard {
    /**
     * game difficulty
     */
    enum Type {EASY, MEDIUM, HARD}

    /**
     * score class
     */
    public class Score implements Comparable {
        /**
         * info about a score
         */
        public String name;
        public String time;
        public Type type;
        private int difficulty;

        /**
         * score constructor
         * @param n name
         * @param t time
         * @param ty type
         */
        public Score(String n, String t, Type ty) {
            name = n;
            time = t;
            type = ty;
            if (type == EASY) difficulty = 1;
            else if (type == MEDIUM) difficulty = 2;
            else difficulty = 3;
        }

        /**
         * info about a score
         * @return the score as a string
         */
        public String toString() {
            return String.format("%-10s %-10s %-10s", name, time, (type == EASY ? "easy" : (type == MEDIUM ? "med" : "hard"))) + ('\n');
        }

        /**
         * how to compare two scores
         * @param o the other score
         * @return the comparison of the scores
         */
        @Override
        public int compareTo(Object o) {
            if (Integer.compare(this.difficulty, ((Score) o).difficulty) != 0)
                return this.difficulty - ((Score) o).difficulty;
            else
                return Integer.parseInt(this.time) - Integer.parseInt(((Score) o).time);
        }
    }

    /**
     * location of the high score file
     */
    private static final String fileName = "./data/highScore.dat";
    /**
     * array of scores
     */
    ArrayList<Score> scores;
    /**
     * max scores to be recorded
     */
    private int maxScores = 10;

    /**
     * score board constructor reads existing score file if possible
     */
    public ScoreBoard() {
        Scanner fileInput = null;
        scores = new ArrayList<Score>();
        try {
            fileInput = new Scanner(new FileReader(new File(fileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Existing directory file not found, creating new directory.");
            return;
        }

        while (fileInput.hasNext()) {
            String name = fileInput.next();
            String time = fileInput.next();
            String difficulty = fileInput.next();
            scores.add(new Score(name, time, (difficulty.equalsIgnoreCase("easy") ? EASY : (difficulty.equalsIgnoreCase("med") ? MEDIUM : HARD))));
        }
    }

    /**
     * the scores as a string
     * @return the scores as a string
     */
    public String toString() {
        String str = new String();
        int  num = 1;
        for (Score s : scores) {
            str += num++;
            str += "\t";
            str += s.toString();
        }
        return str;
    }

    /**
     * clears the scores
     */
    public void clearScores(){
        scores.clear();
    }

    /**
     * creats a new score
     * @param time time of the new score
     * @param type type of the new score
     * @return true if the score was a new high score
     */
    public boolean newHighScore(String time, Type type){
        if(scores.size() == 0 || scores.size() < maxScores || Integer.parseInt(time) < Integer.parseInt(scores.get(scores.size()-1).time)){
            if(scores.size() >= maxScores){
                scores.remove(scores.size()-1);
            }
            TextInputDialog dialog = new TextInputDialog("Name");
            dialog.setTitle("New High Score");
            dialog.setHeaderText(null);
            dialog.setContentText("Name");
            String name = dialog.showAndWait().get();
            dialog.close();

            scores.add(new Score(name, time, type));

            Collections.sort(scores);
            return true;
        }
        else
            return false;
    }

    /**
     * writes the scores to a file
     */
    public void save() {
        PrintWriter fileOutput = null;

        try {
            fileOutput = new PrintWriter(new FileOutputStream(fileName, false));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for (Score s : scores)
            fileOutput.println(s.name + ' ' + s.time + ' ' + (s.type == EASY ? "easy" : (s.type == MEDIUM ? "med" : "hard")));
        fileOutput.close();
    }
}