package cs2410.assn3.gui;

import cs2410.assn3.directory.Directory;
import javafx.application.Application;
import javafx.scene.control.*;
import javafx.scene.layout.Region;
import javafx.stage.Stage;
import javafx.stage.Window;

import java.util.ArrayList;
import java.util.Optional;

/**
 * Created by Philip Nelson on 1/27/2017.
 */
public class GUIDirectory extends Application {

    /**
     * Launces GUI application
     *
     * @param args command line argumetns
     */
    public static void main(String[] args) {
        Application.launch(GUIDirectory.class);
    }

    /**
     * GUI application
     *
     * @param primaryStage
     * @throws Exception
     */
    public void start(Stage primaryStage) throws Exception {
        Directory dir = new Directory();
        Alert alert;
        TextInputDialog dialog;

        while (true) {
            ArrayList<String> list = new ArrayList();
            list.add("List directory");
            list.add("Add student");
            list.add("Display averate age");
            list.add("Quit");

            ChoiceDialog choice = new ChoiceDialog("Quit", list);
            choice.setTitle("Menu:");
            choice.setHeaderText(null);
            choice.setContentText("Choose one");
            Window window = choice.getDialogPane().getScene().getWindow();
            window.setOnCloseRequest(event -> System.exit(1));

            Optional<String> result2 = choice.showAndWait();
            if (result2.isPresent()) {
                if (list.get(0) == result2.get()) {//List Directory
                    alert = new Alert(Alert.AlertType.INFORMATION);
                    alert.setHeaderText(null);
                    alert.setGraphic(null);
                    alert.setTitle("DIRECTORY");
                    String sDir = dir.toString();
                    alert.setContentText(sDir);
                    alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
                    alert.getDialogPane().getStylesheets().add("resources/custom.css");
                    alert.showAndWait();
                }
                if (list.get(1) == result2.get()) {//Add Student
//                    First Name
                    dialog = new TextInputDialog("");
                    dialog.setTitle("New Student");
                    dialog.setHeaderText(null);
                    dialog.setContentText("First Name");
                    String fn = dialog.showAndWait().get();
                    dialog.close();
//                    Last Name
                    dialog = new TextInputDialog("");
                    dialog.setTitle("New Student");
                    dialog.setHeaderText(null);
                    dialog.setContentText("Last Name");
                    String ln = dialog.showAndWait().get();
                    dialog.close();
//                    Age
                    dialog = new TextInputDialog("");
                    dialog.setTitle("Age");
                    dialog.setHeaderText(null);
                    dialog.setContentText("Age");
                    int age = Integer.parseInt(dialog.showAndWait().get());
                    dialog.close();
//                    Phone Numebr
                    dialog = new TextInputDialog("");
                    dialog.setTitle("New Student");
                    dialog.setHeaderText(null);
                    dialog.setContentText("Phone Number");
                    String pn = dialog.showAndWait().get();
                    dialog.close();

//                    Make New Student
                    String newS = dir.addStudent(fn, ln, age, pn);

//                    Display New Student Info
                    alert = new Alert(Alert.AlertType.INFORMATION);
                    alert.setHeaderText(null);
                    alert.setGraphic(null);
                    alert.setTitle("New Student");
                    alert.setContentText(newS);
                    Window win = alert.getDialogPane().getScene().getWindow();
                    window.setOnCloseRequest(event -> win.hide());
                    alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
                    alert.showAndWait();
                    dialog.close();
                }

                if (list.get(2) == result2.get()) {//Display Average
                    alert = new Alert(Alert.AlertType.INFORMATION);
                    alert.setHeaderText(null);
                    alert.setGraphic(null);
                    alert.setTitle("Average Student Age");
                    double ave = dir.getAve();
                    alert.setContentText(String.format("%s %.2f", "The average student age is: ", ave));
                    Window win = alert.getDialogPane().getScene().getWindow();
                    window.setOnCloseRequest(event -> win.hide());
                    alert.showAndWait();
                }

                if (list.get(3) == result2.get()) {//Quit
                    System.exit(1);
                }
            }
        }
    }
}

