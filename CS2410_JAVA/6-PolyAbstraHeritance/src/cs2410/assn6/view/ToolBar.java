package cs2410.assn6.view;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.layout.HBox;

/**
 * Created by philip_nelson on 3/27/17.
 */
public class ToolBar extends HBox {
    /**
     * combo box for math, income, hours, iq, carrots and contracts
     */
    public ComboBox comboBox = new ComboBox();
    /**
     * hourly button
     */
    public Button hourly = new Button("Hourly");
    /**
     * contract button
     */
    public Button contract = new Button("Contract");
    /**
     * hobbit button
     */
    public Button hobbit = new Button("Hobbit");

    /**
     * toolbar constructor
     */
    public ToolBar() {
        comboBox.getItems().addAll( "Math", "Income", "Hours", "IQ", "Say", "Carrots", "Contracts");
//        comboBox.setVisibleRowCount(3);
        this.getChildren().addAll(comboBox, hourly, contract, hobbit);
        this.setPadding(new Insets(5, 5, 5, 5));
        this.setSpacing((5));
        this.setAlignment(Pos.CENTER);
    }
}
