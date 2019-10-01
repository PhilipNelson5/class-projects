package cs2410.assn8.view;

import cs2410.assn8.control.Control;
import cs2410.assn8.control.ScoreBoard;
import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.*;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.stage.Stage;
import javafx.stage.Window;

import java.io.File;

public class View extends Application {


    public static void main(String[] args) {
        launch(args);
    }

    /**
     * animation timer for clock
     */
    AnimationTimer timer;

    /**
     * GUI elements
     */
    Stage pstage;
    Scene center;
    FlowPane grid;
    BorderPane view;
    HBox topBar;
    MenuBar menuBar;
    Menu fileMenu, gameMenu, helpMenu;
    Menu subSize, subDifficulty, subMode;
    MenuItem itemHighScores, itemClearHighScores, itemExit, itemMute;
    MenuItem itemSmall, itemMedium, itemLarge, itemEasy, itemMed, itemHard;
    MenuItem itemNormal, itemSpeedDeamon, itemTimeUp;
    Button start;
    private Label bombsRemaining, gameTime;

    /**
     * data fields
     */
    FieldButton[][] cellList;
    int buttonSize;
    boolean playing = false;
    boolean win = false;
    boolean mute = false;
    long lastUpdate = 0;

    /**
     * all the sounds
     */
    MediaPlayer click = new MediaPlayer(new Media(new File(("./Resources/click.wav")).toURI().toString()));
    MediaPlayer explosion = new MediaPlayer(new Media(new File(("./Resources/explosion.wav")).toURI().toString()));
    MediaPlayer flag = new MediaPlayer(new Media(new File(("./Resources/flag.wav")).toURI().toString()));
    MediaPlayer loseVoice = new MediaPlayer(new Media(new File(("./Resources/loseVoice.wav")).toURI().toString()));
    MediaPlayer winSound = new MediaPlayer(new Media(new File(("./Resources/win.wav")).toURI().toString()));


    /**
     * controllers
     */
    cs2410.assn8.control.Control control;
    cs2410.assn8.control.ScoreBoard scoreBoard;

    @Override
    public void start(Stage primaryStage) throws Exception {
        click.setVolume(1);
        flag.setVolume(2);

        extraCredit();
        playing = false;
        control = new Control();
        scoreBoard = new ScoreBoard();
        pstage = primaryStage;
        buttonSize = 37;

        bombsRemaining = new Label("0");
        gameTime = new Label("0");
        start = new Button("Start");
        start.setPrefSize(65, 30);
        start.setOnAction(event -> start());

        topBar = new HBox();
        topBar.getChildren().addAll(new Label("Bombs Left:"), bombsRemaining, start, new Label("Time:"), gameTime);
        topBar.setAlignment(Pos.CENTER);
        topBar.setSpacing(10);
        topBar.setPadding(new Insets(10, 0, 10, 0));

        itemExit = new MenuItem("Exit");
        itemExit.setGraphic((new ImageView(new Image("file:Resources/exit.png"))));
        itemMute = new MenuItem("Mute");
        itemMute.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemHighScores = new MenuItem("High Scores");
        itemClearHighScores = new MenuItem("Clear High Scores");

        itemEasy = new MenuItem("Easy");
        itemEasy.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemMed = new MenuItem("Medium");
        itemMed.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemHard = new MenuItem("Hard");
        itemHard.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        subDifficulty = new Menu("Difficulty");
        subDifficulty.getItems().addAll(itemEasy, itemMed, itemHard);

        itemSmall = new MenuItem("Small");
        itemSmall.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemMedium = new MenuItem("Medium");
        itemMedium.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemLarge = new MenuItem("Large");
        itemLarge.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        subSize = new Menu("Size");
        subSize.getItems().addAll(itemSmall, itemMedium, itemLarge);

        itemNormal = new MenuItem("Normal");
        itemNormal.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemSpeedDeamon = new MenuItem("Speed Deamon");
        itemSpeedDeamon.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemTimeUp = new MenuItem("Time's UP");
        itemTimeUp.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        subMode = new Menu("Mode");
        subMode.getItems().addAll(itemNormal, itemSpeedDeamon, itemTimeUp);


        fileMenu = new Menu("File");
        fileMenu.getItems().addAll(itemMute, itemExit);

        gameMenu = new Menu("Game");
        gameMenu.getItems().addAll(subDifficulty, subSize, subMode, itemHighScores, itemClearHighScores);

        menuBar = new MenuBar(fileMenu, gameMenu);

        grid = new FlowPane();

        view = new BorderPane();
        view.setTop(topBar);
        Pane blank = new Pane();
        blank.setPrefSize(370, 385);
        view.setCenter(blank);

        BorderPane game = new BorderPane();
        game.setTop(menuBar);
        game.setCenter(view);

        center = new Scene(game);

        primaryStage.setTitle("Minesweeper");
        primaryStage.setResizable(false);
        primaryStage.setOnCloseRequest(event -> quit());
        primaryStage.setScene(center);
        primaryStage.show();

        itemMute.setOnAction(event -> muteSound());
        itemExit.setOnAction(event -> quit());
        itemEasy.setOnAction(event -> setDifficultyEasy());
        itemMed.setOnAction(event -> setDifficultyMedium());
        itemHard.setOnAction(event -> setDifficultyHard());
        itemSmall.setOnAction(event -> setSizeSmall());
        itemMedium.setOnAction(event -> setSizeMedium());
        itemLarge.setOnAction(event -> setSizeLarge());
        itemHighScores.setOnAction(event -> showHighScores());
        itemNormal.setOnAction(event -> setModeNormal());
        itemSpeedDeamon.setOnAction(event -> setModeSpeedDeamon());
        itemTimeUp.setOnAction(event -> setModeTimeUp());
        itemClearHighScores.setOnAction(event -> scoreBoard.clearScores());

        control.bombsRemain.addListener(new ChangeListener<Number>() {
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                bombsRemaining.setText(Integer.toString(control.bombsRemain.get()));
            }
        });

        control.cellsCleared.addListener(new ChangeListener<Number>() {
            @Override
            public void changed(ObservableValue<? extends Number> observable, Number oldValue, Number newValue) {
                if (control.hasWon()) {
                    win = true;
                    endGame();
                }
            }
        });

        view.getStylesheets().add((new File("Resources/FieldButton.css")).toURI().toURL().toExternalForm());
        start();
    }

    /**
     * sets the image for a given cell
     * @param b button to be set
     * @param num number of bombs
     */
    void setImage(Button b, int num) {
        String file;
        switch (num) {
            case 1:
                file = new String("file:Resources/one.png");
                break;
            case 2:
                file = new String("file:Resources/two.png");
                break;
            case 3:
                file = new String("file:Resources/three.png");
                break;
            case 4:
                file = new String("file:Resources/four.png");
                break;
            case 5:
                file = new String("file:Resources/five.png");
                break;
            case 6:
                file = new String("file:Resources/six.png");
                break;
            case 7:
                file = new String("file:Resources/seven.png");
                break;
            default:
                file = new String("file:Resources/eight.png");
                break;
        }
        b.setGraphic(new ImageView(new Image(file)));
    }

    /**
     * rotates through clear, flagges and question
     * @param b
     */
    void switchImage(FieldButton b) {
        String file;
        if (b.status == Status.CLEAR) {
            if (control.bombsRemain.get() == 0) {
                file = new String("file:Resources/question.png");
                b.status = Status.QUESTION;
            } else {
                file = new String("file:Resources/flag.png");
                control.decBombsRemain();
                b.status = Status.FLAG;
            }
        } else if (b.status == Status.FLAG) {
            file = new String("file:Resources/question.png");
            control.incBombsRemain();
            b.status = Status.QUESTION;
        } else {
            file = new String("file:Resources/clear.png");
            b.status = Status.CLEAR;
        }

        b.setGraphic(new ImageView(new Image(file)));
    }

    /**
     * reveals the whole board
     */
    private void revealBoard() {
        for (FieldButton[] row : cellList) {
            for (FieldButton b : row) {
                if (b.type == Type.BOMB) {
                    if (b.status == Status.FLAG) {
                        b.setId("bomb_flagged");
                    } else {
                        b.setGraphic(new ImageView(new Image("file:Resources/bomb.png")));
                        b.setId("bomb_unflagged");
                    }
                } else {
                    if (b.status == Status.FLAG) {
                        b.setId("wrong_flag");
                    }
                }
            }
        }
    }

    /**
     * handles clicking the cells
     * @param row row of cell to be clicked
     * @param col col of cell to be clicked
     */
    private void clickCell(int row, int col) {
        cellList[row][col].cleared = true;
        cellList[row][col].setId("cell_clicked");
        int num = (control.getBombs(cellList[row][col].row, cellList[row][col].col));
        if (num > 0) setImage(cellList[row][col], num);
        cellList[row][col].setDisable(true);
        control.cellsCleared.set(control.cellsCleared.get() + 1);
    }

    /**
     * recursive function to clear all zero cells
     * @param row row of cell
     * @param col col of cell
     */
    private void clearAdj(int row, int col) {
        for (int i = row - 1; i <= row + 1; ++i) {
            for (int j = col - 1; j <= col + 1; ++j) {
                if (i < 0 || j < 0 || i >= control.maxRows || j >= control.maxCols || cellList[i][j].cleared) {
                    continue;
                }
                if (cellList[i][j].type == Type.BOMB) {
                    return;
                }
                if (control.getBombs(i, j) != 0) {
                    clickCell(i, j);
                    continue;
                }
                clickCell(i, j);
                clearAdj(i, j);
            }
        }
    }

    /**
     * cell handler
     * @param b cell to be handled
     */
    private void setCellHandler(FieldButton b) {
        b.setOnMouseClicked(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
                if (!playing) {
                    startTimer();
                    playing = true;
                    start.setDisable(true);
                }
                if (event.getButton() == MouseButton.PRIMARY) {
                    if (b.status == Status.FLAG | b.status == Status.QUESTION)
                        return;
                    if (control.bombMatrix[b.row][b.col]) {
                        if (!mute) {
                            MediaPlayer explosion = new MediaPlayer(new Media(new File(("./Resources/explosion.wav")).toURI().toString()));
                            explosion.play();
                        }
                        b.setGraphic(new ImageView(new Image("file:Resources/explosion.png")));
                        System.out.println("BOOM");
                        endGame();
                    } else {
                        if (!mute) {
                            MediaPlayer click = new MediaPlayer(new Media(new File(("./Resources/click.wav")).toURI().toString()));
                            click.play();
                        }
                        if (control.mode == Control.Mode.SPEED_DEAMON)
                            gameTime.setText(Integer.toString(control.timeInterval));
                        int num = (control.getBombs(b.row, b.col));
                        if (num == 0)
                            clearAdj(b.row, b.col);
                        else
                            clickCell(b.row, b.col);
                    }
                } else if (event.getButton() == MouseButton.SECONDARY) {
                    if (!mute) {
                        MediaPlayer flag = new MediaPlayer(new Media(new File(("./Resources/flag.wav")).toURI().toString()));
                        flag.play();
                    }
                    switchImage(b);

                }
            }

        });

        b.setOnMousePressed(new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent event) {
            }
        });
    }

    /**
     * game winning logic
     */
    void gameWon() {
        if (!mute) {
            MediaPlayer winSound = new MediaPlayer(new Media(new File(("./Resources/win.wav")).toURI().toString()));
            winSound.play();
        }
        System.out.println("YOU WIN!!!");
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setHeaderText(null);
        alert.setGraphic(null);
        alert.setTitle("WINNER!!!");
        alert.setContentText("You cleared the mine field in " + gameTime.getText() + " seconds!");
        alert.showAndWait();

        if (scoreBoard.newHighScore(gameTime.getText(), control.type)) ;
        {
            alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setHeaderText(null);
            alert.setGraphic(null);
            alert.setTitle("High Scores");
            alert.setContentText(scoreBoard.toString());
            Window win = alert.getDialogPane().getScene().getWindow();
            win.setOnCloseRequest(event -> win.hide());
            alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
            alert.showAndWait();
        }
    }

    /**
     * game losing loginc
     */
    void gameLost() {
        if (!mute) {
            MediaPlayer loseVoice = new MediaPlayer(new Media(new File(("./Resources/loseVoice.wav")).toURI().toString()));
            loseVoice.play();
        }
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setHeaderText(null);
        alert.setGraphic(null);
        alert.setTitle("LOSER!!!");
        switch (control.mode) {
            case NORMAL:
                alert.setContentText("You failed to cleared the mine field!");
                break;
            case SPEED_DEAMON:
            case TIME_UP:
                alert.setContentText("You failed to cleared the mine field!\nClick faster next time!");
                break;
        }
        alert.showAndWait();
    }

    /**
     * handles the end of the game
     */
    void endGame() {
        timer.stop();
        revealBoard();
        lockGrid();
        start.setDisable(false);
        start.setText("Reset");
        playing = false;
        if (win)
            gameWon();
        else
            gameLost();
    }

    /**
     * starts the timer
     */
    void startTimer() {//http://stackoverflow.com/questions/30146560/how-to-change-animationtimer-speed
        timer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= 1_000_000_000) {
                    int time = (Integer.parseInt(gameTime.getText()));
                    switch (control.mode) {
                        case NORMAL:
                            gameTime.setText(Integer.toString(++time));
                            lastUpdate = now;
                            break;
                        case SPEED_DEAMON:
                        case TIME_UP:
                            gameTime.setText(Integer.toString(--time));
                            lastUpdate = now;
                            if (time <= 0) {
                                Platform.runLater(new Runnable() {
                                    @Override
                                    public void run() {
                                        endGame();
                                    }
                                });
                            }
                            break;
                    }
                }
            }
        };
        timer.start();
    }

    /**
     * starts the game
     */
    void start() {
        if (playing) timer.stop();
        switch (control.mode) {
            case NORMAL:
                gameTime.setText("0");
                break;
            case SPEED_DEAMON:
                gameTime.setText(Integer.toString(control.timeInterval));
                break;
            case TIME_UP:
                gameTime.setText(Integer.toString(control.maxTime));
                break;
        }
        playing = false;
        win = false;
        start.setText("Start");
        control.initGrid();
        grid = new FlowPane();
        cellList = new FieldButton[control.maxRows][control.maxCols];
        grid.setPrefSize(control.maxCols * buttonSize, control.maxRows * buttonSize);
        for (int i = 0, b = 0; i < control.maxRows; ++i) {
            for (int j = 0; j < control.maxCols; ++j, ++b) {
//                FieldButton temp = new FieldButton(i, j, (control.bombMatrix[i][j] ? Type.BOMB : Type.SPACE), (control.bombMatrix[i][j] ? "B" : ""));
                FieldButton temp = new FieldButton(i, j, (control.bombMatrix[i][j] ? Type.BOMB : Type.SPACE), "");
                temp.setPrefSize(buttonSize, buttonSize);
                temp.setId("cell_unclicked");
                temp.setAlignment(Pos.CENTER);
                temp.setOnAction(event -> System.out.println(control.getBombs(temp.row, temp.col) + " " + temp.row + " " + temp.col));
                grid.getChildren().add(temp);
                cellList[i][j] = temp;
                setCellHandler(temp);
            }
        }
        view.setCenter(grid);
        pstage.sizeToScene();
    }

    /**
     * locks all buttons on the grid
     */
    void lockGrid() {
        if (grid != null) {
            grid.setDisable(true);
        }
    }

    /**
     * shows the high score
     */
    void showHighScores() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setHeaderText(null);
        alert.setGraphic(null);
        alert.setTitle("High Scores");
        alert.setContentText(scoreBoard.toString());
        Window win = alert.getDialogPane().getScene().getWindow();
        win.setOnCloseRequest(event -> win.hide());
        alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
        alert.showAndWait();
    }

    /**
     * sets the game to small size
     */
    void setSizeSmall() {
        itemSmall.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemMedium.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemLarge.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setSizeSmall();
        start();
    }

    /**
     * sets the game to medium size
     */
    void setSizeMedium() {
        itemSmall.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemMedium.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemLarge.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setSizeMedium();
        lockGrid();
        start();
    }

    /**
     * sets the game to large size
     */
    void setSizeLarge() {
        itemSmall.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemMedium.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemLarge.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        control.setSizeLarge();
        start();
    }

    /**
     * sets the game to easy difficulty
     */
    void setDifficultyEasy() {
        itemEasy.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemMed.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemHard.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setDiffEasy();
        start();
    }

    /**
     * sets the game to medium difficulty
     */
    void setDifficultyMedium() {
        itemEasy.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemMed.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemHard.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setDiffMedium();
        start();
    }

    /**
     * sets the game to hard difficulty
     */
    void setDifficultyHard() {
        itemEasy.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemMed.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemHard.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        control.setDiffHard();
        start();
    }

    /**
     * sets game mode to normal
     */
    void setModeNormal() {
        itemNormal.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemSpeedDeamon.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemTimeUp.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setModeNormal();
        start();
    }

    /**
     * sets game mode to Seed Deamon
     */
    void setModeSpeedDeamon() {
        itemNormal.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemSpeedDeamon.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        itemTimeUp.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        control.setModeSpeedDeamon();
        start();

    }

    /**
     * sets game mode to times up
     */
    void setModeTimeUp() {
        itemNormal.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemSpeedDeamon.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
        itemTimeUp.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        control.setModeTimeUp();
        start();

    }

    /**
     * mutes all sounds
     */
    void muteSound() {
        mute = !mute;
        if (mute)
            itemMute.setGraphic((new ImageView(new Image("file:Resources/checked.png"))));
        else
            itemMute.setGraphic((new ImageView(new Image("file:Resources/unchecked.png"))));
    }

    /**
     * handles the cleanup on quit
     */
    void quit() {
        scoreBoard.save();
        Platform.exit();
    }

    /**
     * Extra Credit dialog
     */
    void extraCredit() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setHeaderText(null);
        alert.setGraphic(null);
        alert.setTitle("Extra Credit");
        alert.setContentText("I have successfully implemented all the extra credit features.\n" +
                "As it is stated in the instructions I deserve:\n" +
                "10 points - size feature\n" +
                "10 points - difficulty feature\n" +
                " 5 points - high score feature\n" +
                "15 points - sounds\n" +
                "20 points - modes: Speed Daemon & Times Up\n" +
                "------------------------------------------------------------------\n" +
                "60 points total - Thank you for grading all semester :D \n\n" +
                "All other features for the assignment are also implemented.");
        Window win = alert.getDialogPane().getScene().getWindow();
        win.setOnCloseRequest(event -> win.hide());
        alert.getDialogPane().getChildren().stream().filter(node -> node instanceof Label).forEach(node -> ((Label) node).setMinHeight(Region.USE_PREF_SIZE));
        alert.showAndWait();
    }
}