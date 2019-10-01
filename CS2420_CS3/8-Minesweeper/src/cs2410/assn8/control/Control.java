package cs2410.assn8.control;

import javafx.beans.property.SimpleIntegerProperty;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by philip_nelson on 4/25/17.
 */
public class Control {

    /**
     * game mode
     */
    public enum Mode{NORMAL, SPEED_DEAMON, TIME_UP}

    /**
     * max values
     */
    public int maxRows, maxCols, maxBombs, maxTime, timeInterval;

    /**
     * easy, medium or hard
     */
    public ScoreBoard.Type type;

    /**
     * game mode
     */
    public Mode mode;

    /**
     * bomb info
     */
    public SimpleIntegerProperty bombsRemain, cellsCleared;
    public double percentBombs;
    public boolean bombMatrix[][];

    /**
     * constructor, defaults to small, easy, normal
     */
    public Control() {
        maxRows = 0;
        maxCols = 0;
        maxBombs = 0;
        bombsRemain = new SimpleIntegerProperty(0);
        cellsCleared = new SimpleIntegerProperty(0);
        setDiffEasy();
        setSizeSmall();
        setModeNormal();
    }

    /**
     * returns the number of bombs around a give cell
     * @param row row of cell
     * @param col col of cell
     * @return number of adjacent bombs
     */
    public int getBombs(int row, int col) {
        int bombs = 0;
        for (int i = row - 1; i <= row + 1; ++i)
            for (int j = col - 1; j <= col + 1; ++j)
                if (i >= 0 && j >= 0 && i < maxRows && j < maxCols && bombMatrix[i][j])
                    ++bombs;
        return bombs;

    }

    /**
     * handles decrementing the bomb count
     */
    public void decBombsRemain() {
        if (bombsRemain.get() == 0)
            return;
        bombsRemain.set(bombsRemain.get() - 1);
    }

    /**
     * handles incrementing the bomb count
     */
    public void incBombsRemain() {
        if (bombsRemain.get() == maxBombs)
            return;
        bombsRemain.set(bombsRemain.get() + 1);
    }

    /**
     * initializes the cells with current params
     */
    public void initGrid() {
        bombsRemain.set((int) (maxRows * maxCols * percentBombs));
        cellsCleared.set(0);
        maxBombs = (int) (maxRows * maxCols * percentBombs);
        bombMatrix = new boolean[maxRows][maxCols];
        ArrayList<Boolean> bombList = new ArrayList<>();

        int bombs = 0;
        for (int i = 0; i < maxRows; ++i) {
            for (int j = 0; j < maxCols; ++j) {
                if (bombs++ < maxBombs)
                    bombList.add(true);
                else
                    bombList.add(false);
            }
        }
        Collections.shuffle(bombList);

        for (int i = 0, b = 0; i < maxRows; ++i) {
            for (int j = 0; j < maxCols; ++j, ++b) {
                bombMatrix[i][j] = (bombList.get(b));
            }
        }
    }

    /**
     * checcks win condition
     * @return true if the player has won
     */
    public boolean hasWon() {
        return (cellsCleared.get() + maxBombs) == (maxRows * maxCols);
    }


    /**
     * sets difficulty to easy
     */
    public void setDiffEasy() {
        type = ScoreBoard.Type.EASY;
        maxTime = 10000;
        timeInterval = 100;
        percentBombs = 0.1;
    }

    /**
     * sets difficulty to medium
     */
    public void setDiffMedium() {
        percentBombs = 0.25;
        type = ScoreBoard.Type.MEDIUM;
        maxTime = 1000;
        timeInterval = 25;
    }

    /**
     * sets difficulty to hard
     */
    public void setDiffHard() {
        percentBombs = 0.4;
        type = ScoreBoard.Type.HARD;
        maxTime = 100;
        timeInterval = 10;
    }

    /**
     * sets size to small
     */
    public void setSizeSmall() {
        maxRows = 10;
        maxCols = 10;
    }

    /**
     * sets size to medium
     */
    public void setSizeMedium() {
        maxRows = 25;
        maxCols = 25;
    }

    /**
     * sets size to large
     */
    public void setSizeLarge() {
        maxRows = 25;
        maxCols = 50;
    }

    /**
     * sets mode to normal
     */
    public void setModeNormal(){
        mode = Mode.NORMAL;
    }

    /**
     * sets mode to speed deamon
     */
    public void setModeSpeedDeamon(){
        mode = Mode.SPEED_DEAMON;
    }

    /**
     * sets mode to times up
     */
    public void setModeTimeUp(){
        mode=Mode.TIME_UP;
    }
}