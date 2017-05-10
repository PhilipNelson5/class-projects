package cs2410.assn6.view;

import javafx.scene.control.Alert;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class DelegateHobbit implements Delegate{
    /**
     * saves a hobbit
     * @param cur the gui
     */
    @Override
    public void save(View cur) {
        cur.control.createHobbit(cur.name.getText(),cur.math.getText(),cur.say.getText(),Integer.parseInt(cur.carrots.getText()));

        String confirm = new String(cur.name.getText() + " the hobbit was added!");

        cur.name.clear();
        cur.math.clear();
        cur.say.clear();
        cur.carrots.clear();

        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setHeaderText("Added new worker...");
        info.setContentText(confirm);

        info.showAndWait();
    }
}
