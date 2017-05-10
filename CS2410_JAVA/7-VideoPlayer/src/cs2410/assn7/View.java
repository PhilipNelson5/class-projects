package cs2410.assn7;
// icons by: http://www.flaticon.com/authors/madebyoliver

import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.scene.web.WebView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.io.File;

/**
 * Created by philip_nelson on 4/11/17.
 */
public class View extends Application {
    /**
     * menu bar things
     */
    private MenuBar menuBar;
    private Menu fileMenu, editMenu, helpMenu;
    private MenuItem itemOpen, itemOpenURL, itemClose, itemExit, itemDoc, itemAbout, itemPlayPause, itemStop;
    private MediaPlayer player;
    private MediaView mediaView;
    private Media video;
    /**
     * file chooser things
     */
    private FileChooser fileBrowser;
    private final FileChooser fileChooser = new FileChooser();
    /**
     * buttons
     */
    private Button play_pause, stop;
    /**
     * sliders
     */
    private Slider volume, timeline;
    /**
     * max volume
     */
    private static final float MAX_VOL = 100;
    /**
     * playing flag
     */
    private boolean playing = false;
    /**
     * base border pane
     */
    private BorderPane baseBorderPane = new BorderPane();
    /**
     * webview
     */
    private WebView webview;

    @Override
    /**
     * launch application
     */
    public void start(Stage primaryStage) throws Exception {
        // Menu
        menuBar = new MenuBar();
        fileMenu = new Menu("File");
        fileMenu.getItems().addAll(itemOpen = new MenuItem("Open"), itemClose = new MenuItem("Close"), itemExit = new MenuItem("Exit"));
        itemOpenURL = new MenuItem("Open from URL");
//        fileMenu.getItems().add( itemOpenURL);
        itemOpen.setGraphic((new ImageView(new Image("file:Resources/file.png"))));
        itemOpenURL.setGraphic((new ImageView(new Image("file:Resources/internet.png"))));
        itemClose.setGraphic((new ImageView(new Image("file:Resources/close.png"))));
        itemExit.setGraphic((new ImageView(new Image("file:Resources/exit.png"))));
//        itemOpen.setGraphic();
        editMenu = new Menu("Edit");
        editMenu.getItems().addAll(itemPlayPause = new MenuItem("Play/Pause"), itemStop = new MenuItem("Stop"));
        itemPlayPause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
        itemStop.setGraphic((new ImageView(new Image("file:Resources/stop.png"))));
        helpMenu = new Menu("Help");
        helpMenu.getItems().addAll(itemDoc = new MenuItem("Documentation"), itemAbout = new MenuItem("About"));
        itemDoc.setGraphic((new ImageView(new Image("file:Resources/docs.png"))));
        itemAbout.setGraphic((new ImageView(new Image("file:Resources/info.png"))));
        menuBar.getMenus().addAll(fileMenu, editMenu, helpMenu);

        // File Chooser
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("all file types", "*.*"),
                new FileChooser.ExtensionFilter("MP4(.mp4)", "*.mp4"),
                new FileChooser.ExtensionFilter("M4V(.m4v)", "*.m4v"),
                new FileChooser.ExtensionFilter("M4A(.m4a)", "*.4a4")
        );

        // Controls
        play_pause = new Button();
        play_pause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
        play_pause.setDisable(true);
        stop = new Button();
        stop.setGraphic((new ImageView(new Image("file:Resources/stop.png"))));
        stop.setDisable(true);
        volume = new Slider();
        volume.setMax(MAX_VOL);
        volume.setMin(0);
        volume.setValue(MAX_VOL / 2);
        timeline = new Slider();
        timeline.setMin(0);
        HBox controls = new HBox();
        controls.getChildren().addAll(play_pause, stop, new Label("Volume:"), volume, new Label("Time:"), timeline);
        controls.setAlignment(Pos.CENTER);
        controls.setSpacing(30);

        // File Chooser
        fileChooser.setTitle("Choose Video");

        // Border Pane
        baseBorderPane.setTop(menuBar);
        baseBorderPane.setBottom(controls);
        baseBorderPane.setPadding(new Insets(0, 0, 0, 0));

        // Primary Stage
        Scene scene = new Scene(baseBorderPane, 700, 700);
        primaryStage.setTitle("Video Player");
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(e -> Platform.exit());
        primaryStage.show();

        primaryStage.heightProperty().addListener(new ChangeListener<Number>() {
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if (player != null) {
                    mediaView.setFitHeight(baseBorderPane.getHeight() - 100);
                }
            }
        });

        primaryStage.widthProperty().addListener(new ChangeListener<Number>() {
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if (player != null) {
                    mediaView.setFitWidth(baseBorderPane.getWidth());
                }
            }
        });

        play_pause.setOnAction(event -> togglePlayPause());

        stop.setOnAction(event -> stopVideo());

        itemOpen.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                File file = fileChooser.showOpenDialog(primaryStage);
                if (file == null) return;
                if (player != null)
                    player.dispose();
                if (webview != null) webview = null;
                video = new Media(file.toURI().toString());
                player = new MediaPlayer(video);
                player.setOnReady(new Runnable() {
                    @Override
                    public void run() {
                        mediaView = new MediaView(player);
                        mediaView.setFitWidth(baseBorderPane.getWidth());
                        mediaView.setFitHeight(baseBorderPane.getHeight() - 100);
                        baseBorderPane.setCenter(mediaView);
                        play_pause.setDisable(false);
                        stop.setDisable(false);
                        timeline.setMax(video.getDuration().toMillis());
                        timeline.setValue(0);
                        player.setVolume(volume.getValue() / 100);
                        player.currentTimeProperty().addListener(new ChangeListener<Duration>() {
                            @Override
                            public void changed(ObservableValue<? extends Duration> observable, Duration oldValue, Duration newValue) {
                                timeline.setValue(player.getCurrentTime().toMillis());
                            }
                        });
                    }
                });
            }
        });

        itemOpenURL.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (player != null) player.dispose();
                TextInputDialog dialog;
                dialog = new TextInputDialog("");
                dialog.setTitle("Enter web URL");
                dialog.setHeaderText(null);
                dialog.setContentText("URL: ");
                String url = dialog.showAndWait().get();
                webview = new WebView();
                webview.getEngine().load(url);
                webview.setPrefSize(640, 390);
                baseBorderPane.setCenter((webview));
                dialog.close();
            }
        });

        itemClose.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (player != null) {
                    player.dispose();
                    play_pause.setDisable(true);
                    stop.setDisable(true);
                }
            }
        });

        itemExit.setOnAction(event -> Platform.exit());

        itemPlayPause.setOnAction(event -> togglePlayPause());

        itemStop.setOnAction(event -> stopVideo());

        itemAbout.setOnAction(event -> dispAbout());

        itemDoc.setOnAction(event -> dispDoc());

        volume.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (player != null) {
                    player.setVolume(volume.getValue() / 100);
                }
            }
        });

        timeline.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (player != null) {
                    player.seek(new Duration(timeline.getValue()));
                }
            }
        });

        scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
            @Override
            public void handle(KeyEvent event) {
                if (event.getCode() == KeyCode.SPACE)
                    togglePlayPause();
                if (event.getCode() == KeyCode.P)
                    togglePlayPause();
            }
        });
    }

    /**
     * toggles play and pause
     */
    void togglePlayPause() {
        if (player != null) {
            if (playing) {// pause
                play_pause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
                itemPlayPause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
                player.pause();
            } else {// play
                play_pause.setGraphic((new ImageView(new Image("file:Resources/pause.png"))));
                itemPlayPause.setGraphic((new ImageView(new Image("file:Resources/pause.png"))));
                player.play();
            }
            playing = !playing;
        }
    }

    /**
     * stops the video
     */
    void stopVideo() {
        if (player != null) {
            player.stop();
            playing = false;
            play_pause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
            itemPlayPause.setGraphic((new ImageView(new Image("file:Resources/play.png"))));
            timeline.setValue(0);
        }
    }

    /**
     * displays the about window
     */
    void dispAbout() {
        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setHeaderText("About Video Player");
        info.setContentText("icons by: http://www.flaticon.com/authors/madebyoliver\n" +
                "developed by: Philip Nelson");
        info.showAndWait();
    }

    /**
     * displays the docs window
     */
    void dispDoc() {
        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setHeaderText("Documentation");
        info.setContentText("A simple video player application. It has basic functionality to play, pause, stop, and seek through a video.");
        info.showAndWait();
    }

}
