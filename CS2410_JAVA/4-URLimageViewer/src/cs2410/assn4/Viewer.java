package cs2410.assn4;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextInputDialog;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;

/**
 * Created by philip_nelson on 2/13/17.
 */
public class Viewer extends Application implements EventHandler<ActionEvent> {
    private Scene scene1;
    private Button btnNext, btnPrev, btnAdd, btndDel;
    private Pane pane1;
    private Image image1;
    private EventHandler<ActionEvent> handler;
    private ImageView view1;
    private static final int hight = 700;
    private static final int width = 700;

    /**
     * image URL manager
     */
    private Controller images;

    /**
     * main method
     * @param args command line arguments
     */
    public static void main(String[] args) {
        Application.launch(Viewer.class);
    }

    @Override
    public void start(Stage primaryStage) {
        images = new Controller();

        pane1 = new Pane();

        scene1 = new Scene(pane1, width, hight);

        primaryStage.setScene(scene1);

        btnPrev = new Button("<- Prev");
        btnPrev.setPrefWidth(100);
        btnPrev.setLayoutX(150);
        btnPrev.setLayoutY(hight-30);
        btnPrev.setOnAction(this);

        btnNext = new Button("Next ->");
        btnNext.setPrefWidth(100);
        btnNext.setLayoutX(250);
        btnNext.setLayoutY(hight-30);
        btnNext.setOnAction(this);

        btnAdd = new Button("Add");
        btnAdd.setPrefWidth(100);
        btnAdd.setLayoutX(350);
        btnAdd.setLayoutY(hight-30);
        btnAdd.setOnAction(this);

        btndDel = new Button("Delete");
        btndDel.setPrefWidth(100);
        btndDel.setLayoutX(450);
        btndDel.setLayoutY(hight-30);
        btndDel.setOnAction(this);


        view1 = new ImageView();
        view1.setImage(currImage());

        pane1.getChildren().add(view1);
        pane1.getChildren().addAll(btnPrev, btnNext, btnAdd, btndDel);

        primaryStage.setResizable(false);
        primaryStage.show();
        primaryStage.setOnCloseRequest(event -> {
            images.quit();
        });
    }

    /**
     * prompts user for a URL and adds the image to the viewer
     *
     * @return New image to be displayed
     */
    private Image addNewImage() {
        TextInputDialog dialog;
        dialog = new TextInputDialog("");
        dialog.setTitle("Add Image");
        dialog.setHeaderText(null);
        dialog.setContentText("URL");
        String url = dialog.showAndWait().get();
        dialog.close();
        images.add(url);
        if (btnNext.isDisabled()) {
            btnNext.setDisable(false);
            btnPrev.setDisable(false);
            btndDel.setDisable(false);
        }
        return new Image(images.getNext(), width, hight, true, true);
    }

    /**
     * removes the current image
     * disables buttons when necessary
     *
     * @return New image to be displayed
     */
    private Image deleteImage() {
        images.deleteCurr();
        if (images.size() != 0) {
            return new Image(images.getNext(), width, hight, true, true);
        } else {
            btnNext.setDisable(true);
            btnPrev.setDisable(true);
            btndDel.setDisable(true);
            return new Image("file:data/notfound.jpg", width, hight, true, true);
        }
    }

    /**
     * gets the current image
     *
     * @return current image
     */
    private Image currImage() {
        if (images.size() != 0) {
            return new Image(images.getCurr(), width, hight, true, true);
        } else {
            btnNext.setDisable(true);
            btnPrev.setDisable(true);
            btndDel.setDisable(true);
            return new Image("file:data/notfound.jpg", width, hight, true, true);
        }
    }

    /**
     * gets the next image
     *
     * @return next image
     */
    private Image nextImage() {
        return new Image(images.getNext(), width, hight, true, true);
    }

    /**
     * gets the previous image
     *
     * @return next image
     */
    private Image prevImage() {
        return new Image(images.getPrev(), width, hight, true, true);
    }

    /**
     * handles button press events
     *
     * @param event the event to be handled
     */
    public void handle(ActionEvent event) {
        if (event.getSource() == btnNext) {
            view1.setImage(nextImage());
        }
        if (event.getSource() == btnPrev) {
            view1.setImage(prevImage());
        }
        if (event.getSource() == btnAdd) {
            view1.setImage(addNewImage());
        }
        if (event.getSource() == btndDel) {
            view1.setImage(deleteImage());
        }
    }
}
