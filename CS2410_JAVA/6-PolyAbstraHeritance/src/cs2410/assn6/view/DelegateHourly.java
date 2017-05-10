package cs2410.assn6.view;

import javafx.scene.control.Alert;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class DelegateHourly implements Delegate{
    /**
     * saves an hourly worker
     * @param cur the gui
     */
    @Override
    public void save(View cur) {
        cur.control.createHourly(cur.name.getText(),cur.math.getText(),cur.say.getText(),cur.iq.getText(),Double.parseDouble(cur.hours.getText()),Double.parseDouble(cur.wage.getText()));

        String confirm = new String(cur.name.getText() + " the hourly worker was added!");

        cur.name.clear();
        cur.math.clear();
        cur.say.clear();
        cur.iq.clear();
        cur.hours.clear();
        cur.wage.clear();

        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setHeaderText("Added new worker...");
        info.setContentText(confirm);

        info.showAndWait();
    }
}
