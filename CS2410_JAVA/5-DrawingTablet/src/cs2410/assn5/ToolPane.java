package cs2410.assn5;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;

import java.util.Optional;

/**
 * Helper Class file for Assignment #6. You can use this without any modification. It's the file I used to
 * create the program in the demo video.
 *
 * @author Chad
 * @version 2.0
 */
public class ToolPane extends HBox {
    private Text fillText = new Text("Fill");
    public final ColorPicker fillPicker = new ColorPicker();
    private Text strokeText = new Text("Stroke");
    public final ColorPicker strokePicker = new ColorPicker();
    //This ComboBox contains three Integer objects with values 1, 3, and 5
    public ComboBox<Integer> strokeSize = new ComboBox<>(FXCollections.observableArrayList(1, 3, 5, 7, 9));
    private ToggleButton editBtn = new ToggleButton("Edit");
    private ToggleButton eraseBtn = new ToggleButton("Erase");
    private ToggleButton ellBtn = new ToggleButton("Ellipse");
    private ToggleButton rectBtn = new ToggleButton("Rectangle");
    private ToggleButton freeBtn = new ToggleButton("Freehand");
    public ToggleButton rainbow = new ToggleButton("More FUN");
    public Button btnUndo = new Button("Re-Draw");

    private boolean warning = true;
    private Alert alert;

    public ToolPane() {
        rainbow.setOnAction(event -> handleRainbow());

        this.getChildren().addAll(fillText, fillPicker, strokeText, strokePicker, strokeSize, editBtn, eraseBtn, btnUndo,
                ellBtn, rectBtn, freeBtn, rainbow);

        ToggleGroup toggleGroup = new ToggleGroup();

        //adding ToggleButtons to a ToggleGroup makes it so only one can be selected at a time.
        toggleGroup.getToggles().addAll(editBtn, eraseBtn,
                ellBtn, rectBtn, freeBtn);
        ellBtn.setSelected(true);

        fillPicker.setValue(Color.WHITE);
        strokePicker.setValue(Color.BLACK);
        fillPicker.setStyle("-fx-color-label-visible: false");
        strokePicker.setStyle("-fx-color-label-visible: false");
        strokeSize.setValue(3);

        this.setPadding(new Insets(5, 5, 5, 5));
        this.setSpacing(5);
    }

    private void handleRainbow() {
        if (warning) {
            alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("WARNING");
            alert.setHeaderText(null);
            alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
            alert.setContentText("This option will create violent flashes of color." +
                    " Those who suffer from epilepsy may not wish to use this functionality." +
                    " Precede at your own risk...");
            Optional result = alert.showAndWait();
            if(result.get() == ButtonType.CANCEL)
                rainbow.setSelected(false);
            else
                warning = false;
        }
    }

//    public void setFillPickerAction(EventHandler<ActionEvent> event) {
//        fillPicker.setOnAction(event);
//    }

    public void setFillPickerValue(Color color) {
        fillPicker.setValue(color);
    }

    public Color getFillPickerValue() {
        return fillPicker.getValue();
    }

//    public void setStrokePickerAction(EventHandler<ActionEvent> event) {
//        strokePicker.setOnAction(event);
//    }

    public void setStrokePickerValue(Color color) {
        strokePicker.setValue(color);
    }

    public Color getStrokePickerValue() {
        return strokePicker.getValue();
    }

//    public void setStrokeSizeAction(EventHandler<ActionEvent> event) {
//        strokeSize.setOnAction(event);
//    }

    public void setStrokeSizeValue(Integer val) {
        strokeSize.setValue(val);
    }

    public Integer getStrokeSizeValue() {
        return strokeSize.getValue();
    }

    public boolean editBtnSelected() {
        return editBtn.isSelected();
    }

    public boolean eraseBtnSelected() {
        return eraseBtn.isSelected();
    }

    public boolean ellBtnSelected() {
        return ellBtn.isSelected();
    }

    public boolean rectBtnSelected() {
        return rectBtn.isSelected();
    }

    public boolean freeBtnSelected() {
        return freeBtn.isSelected();
    }

    public boolean rainBtnSelected() {
        return rainbow.isSelected();
    }
}
