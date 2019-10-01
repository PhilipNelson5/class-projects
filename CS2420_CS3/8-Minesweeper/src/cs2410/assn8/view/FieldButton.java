package cs2410.assn8.view;

import javafx.scene.control.Button;

import static cs2410.assn8.view.Status.CLEAR;

/**
 * Created by philip_nelson on 4/24/17.
 */

/**
 * bomb or not bomb
 */
enum Type{BOMB, SPACE}

/**
 * status of a space cell
 */
enum Status{CLEAR, FLAG, QUESTION}

public class FieldButton extends Button {
    /**
     * info about a field button
     */
    public int row, col;
    Type type;
    Status status;
    boolean cleared;

    /**
     * coppy constructor
     * @param old the button to be coppied
     */
    FieldButton(FieldButton old){
        row = old.row;
        col = old.col;
        type = old.type;
        this.setText(old.getText());
        status = old.status;
        cleared = old.cleared;
    }

    /**
     * parameterize constructor
     * @param r row
     * @param c col
     * @param t type
     * @param s string
     */
    FieldButton(int r, int c, Type t, String s){
        row = r;
        col = c;
        type = t;
        this.setText(s);
        this.status = CLEAR;
        this.cleared = false;
    }
}