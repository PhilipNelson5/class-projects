package cs2410.assn5;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;

import static java.lang.Math.abs;

/**
 * Created by philip_nelson on 2/23/17.
 */

public class DrawingTablet extends Application {
    ToolPane toolPane;
    private Thread thread = null;
    private ColorSpinner runnable = null;
    private Pane drawPane = new Pane();
    private Pane basePane = new Pane();
    private double origX;
    private double origY;
    private double orgSceneX;
    private double orgSceneY;
    private double orgTranslateX;
    private double orgTranslateY;

    private Rectangle drawingSpace;
    private Random rand = new Random();
    private ArrayList<Shape> history;
    MediaPlayer mediaPlayer;

    @Override
    /**
     * sets up the primary stage and initiates the drawing tablet
     */
    public void start(Stage primaryStage) throws Exception {
        String running = "running.mp3";
        Media hit = new Media(new File(running).toURI().toString());
        mediaPlayer = new MediaPlayer(hit);
        mediaPlayer.setStartTime(Duration.seconds(25.5));
        mediaPlayer.setCycleCount(MediaPlayer.INDEFINITE);

        history = new ArrayList<Shape>();

        final int offset = 16;
        final int WIDTH = 760;
        final int HIGHT = 600;

        primaryStage.setTitle("Drawing Tablet");
        Scene mainScene = new Scene(basePane);
        primaryStage.setScene(mainScene);

        toolPane = new ToolPane();
        toolPane.setLayoutX(0);
        toolPane.setLayoutY(0);

        basePane.setPrefSize(WIDTH, HIGHT);

        drawPane.setPrefSize(WIDTH, HIGHT - offset);
        drawPane.setLayoutX(0);
        drawPane.setLayoutY(offset);

        drawingSpace = new Rectangle(WIDTH, HIGHT - offset);
        drawingSpace.setX(0);
        drawingSpace.setY(offset);
        drawingSpace.setFill(Color.WHITE);

        drawPane.setClip(drawingSpace);
        setDrawPaneHandler(drawPane);

        basePane.getChildren().addAll(drawPane, toolPane);

        primaryStage.show();

        toolPane.btnUndo.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (!history.isEmpty())
                    drawPane.getChildren().add(history.remove(history.size() - 1));
            }
        });

        toolPane.fillPicker.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (toolPane.editBtnSelected()) {
                    updateShape();
                }
            }
        });

        toolPane.strokePicker.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (toolPane.editBtnSelected()) {
                    updateShape();
                }
            }
        });

        toolPane.strokeSize.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (toolPane.editBtnSelected()) {
                    updateShape();
                }
            }
        });
    }

    /**
     * sets the fill, stroke and stroke width of a shape
     *
     * @param shape the shape to have its values changed
     */
    private void setShapeSettings(Shape shape) {
        if (shape instanceof Path)
            shape.setFill(null);
        else
            shape.setFill(toolPane.getFillPickerValue());
        shape.setStroke(toolPane.getStrokePickerValue());
        shape.setStrokeWidth(toolPane.getStrokeSizeValue());
    }

    /**
     * starts the thread that changes the shape color
     * starts the music
     */
    private void moreFun() {
        runnable = new ColorSpinner(last);
        thread = new Thread(runnable);
        thread.start();
        mediaPlayer.play();
    }

    /**
     * terminates thread
     * pauses music
     */
    private void endFun() {
        mediaPlayer.pause();
        runnable.terminate();
    }

    /**
     * changes the settings of the last used shape
     */
    public void updateShape() {
        setShapeSettings(last);
    }

    /* ******************************************************************/
    /*                             SHAPE HANDLER                        */
    /* ******************************************************************/

    /**
     * handles the events related to shapes
     *
     * @param shape shape to be handled
     */
    private void setShapeHandler(Shape shape) {
        /* ********************************************************/
        /*                        PRESSED                         */
        /* ********************************************************/
        shape.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                origX = event.getX();
                origY = event.getY();
                orgSceneX = event.getSceneX();
                orgSceneY = event.getSceneY();
                orgTranslateX = shape.getTranslateX();
                orgTranslateY = shape.getTranslateY();

                last = shape;

                if (toolPane.editBtnSelected()) {
                    toolPane.setFillPickerValue((Color) shape.getFill());
                    toolPane.setStrokePickerValue((Color) shape.getStroke());
                    toolPane.setStrokeSizeValue((int) shape.getStrokeWidth());
                    setShapeSettings(shape);
                    drawPane.getChildren().remove(shape);
                    drawPane.getChildren().add(shape);
                }

                if (toolPane.eraseBtnSelected()) {
                    history.add(shape);
                    drawPane.getChildren().remove(shape);
                }

                if (toolPane.rainBtnSelected() && toolPane.editBtnSelected()) {
                    moreFun();
                }
            }
        });

        /* ********************************************************/
        /*                        DRAGGED                         */
        /* ********************************************************/
        shape.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (toolPane.editBtnSelected()) {
                    double offsetX = event.getSceneX() - orgSceneX;
                    double offsetY = event.getSceneY() - orgSceneY;
                    double newTranslateX = orgTranslateX + offsetX;
                    double newTranslateY = orgTranslateY + offsetY;
                    shape.setTranslateX(newTranslateX);
                    shape.setTranslateY(newTranslateY);
                }
            }
        });

        shape.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (toolPane.rainBtnSelected() && !toolPane.eraseBtnSelected())
                    endFun();
            }
        });
    }

    private Ellipse ell;
    private Rectangle rec;
    private Path path;
    private Shape last;

    /* ******************************************************************/
    /*                          DRAW PANE HANDLER                       */
    /* ******************************************************************/

    /**
     * handles the events related to the drawing pane
     *
     * @param drawPane pane to be handled
     */
    private void setDrawPaneHandler(Pane drawPane) {
        /* ********************************************************/
        /*                        PRESSED                         */
        /* ********************************************************/
        drawPane.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                origX = event.getX();
                origY = event.getY();

                /* *******************************************/
                /*                Ellipse                    */
                /* *******************************************/
                if (toolPane.ellBtnSelected()) {
                    ell = new Ellipse();
                    last = ell;
                    setShapeHandler(ell);
                    setShapeSettings(ell);
                    ell.setStrokeType(StrokeType.OUTSIDE);
                    ell.setCenterX(event.getX());
                    ell.setCenterY(event.getY());
                    drawPane.getChildren().add(ell);
                }

                /* *******************************************/
                /*                 Rectangle                 */
                /* *******************************************/
                if (toolPane.rectBtnSelected()) {
                    rec = new Rectangle();
                    last = rec;
                    setShapeHandler(rec);
                    setShapeSettings(rec);
                    rec.setStrokeType(StrokeType.OUTSIDE);
                    rec.setX(event.getX());
                    rec.setY(event.getY());
                    drawPane.getChildren().add(rec);
                }

                /* *******************************************/
                /*                    Line                   */
                /* *******************************************/
                if (toolPane.freeBtnSelected()) {
                    path = new Path();
                    last = path;
                    setShapeHandler(path);
                    drawPane.getChildren().add(path);
                    setShapeSettings(path);
                    path.getElements().add(new MoveTo(event.getX(), event.getY()));
                }

                if (toolPane.rainBtnSelected() && !toolPane.eraseBtnSelected() && !toolPane.editBtnSelected())
                    moreFun();
            }
        });

        /* ********************************************************/
        /*                        DRAGGED                         */
        /* ********************************************************/
        drawPane.setOnMouseDragged(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                /* *******************************************/
                /*                Ellipse                    */
                /* *******************************************/
                if (toolPane.ellBtnSelected()) {
                    double offsetX = (event.getX() - origX);
                    double offsetY = (event.getY() - origY);

                    ell.setRadiusX(abs(offsetX / 2));
                    ell.setRadiusY(abs(offsetY / 2));

                    ell.setCenterX(event.getX() - (offsetX / 1.5));
                    ell.setCenterY(event.getY() - (offsetY / 1.5));
                }

                /* *******************************************/
                /*                 Rectangle                 */
                /* *******************************************/
                if (toolPane.rectBtnSelected()) {
                    double width = event.getX() - origX;
                    double hight = event.getY() - origY;
                    if (width < 0)
                        rec.setX(event.getX());

                    if (hight < 0)
                        rec.setY(event.getY());

                    rec.setWidth(abs(width));
                    rec.setHeight(abs(hight));
                }

                /* *******************************************/
                /*                    Line                   */
                /* *******************************************/
                if (toolPane.freeBtnSelected()) {
                    path.getElements().add(new LineTo(event.getX(), event.getY()));
                }
            }
        });

        drawPane.setOnMouseReleased(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (toolPane.rainBtnSelected() && !toolPane.eraseBtnSelected())
                    endFun();
            }
        });
    }
}
