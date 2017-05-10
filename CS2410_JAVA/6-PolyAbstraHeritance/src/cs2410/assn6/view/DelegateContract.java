package cs2410.assn6.view;

import javafx.scene.control.Alert;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class DelegateContract implements Delegate{
    /**
     * saves a contract worker
     * @param cur the gui
     */
    @Override
    public void save(View cur) {
        cur.control.createContract(cur.name.getText(),cur.math.getText(),cur.say.getText(),cur.iq.getText(),Double.parseDouble(cur.contracts.getText()),Double.parseDouble(cur.payPerContract.getText()));

        String confirm = new String(cur.name.getText() + " the contract worker was added!");

        cur.name.clear();
        cur.math.clear();
        cur.say.clear();
        cur.iq.clear();
        cur.contracts.clear();
        cur.payPerContract.clear();

        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setHeaderText("Added new worker...");
        info.setContentText(confirm);

        info.showAndWait();
    }
}
