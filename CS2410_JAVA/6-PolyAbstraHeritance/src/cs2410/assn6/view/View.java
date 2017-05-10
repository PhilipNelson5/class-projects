package cs2410.assn6.view;

import cs2410.assn6.control.Control;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;

/**
 * Created by philip_nelson on 3/27/17.
 */
public class View extends Application {
    /**
     * delegators
     */
    private Delegate delegator;
    private DelegateContract delegateContract = new DelegateContract();
    private DelegateHourly delegateHourly = new DelegateHourly();
    private DelegateHobbit delegateHobbit = new DelegateHobbit();

    /**
     * border panes
     */
    private BorderPane baseBorderPane = new BorderPane();
    private BorderPane hourlyBorderPane = new BorderPane();
    private BorderPane contractBorderPane = new BorderPane();
    private BorderPane hobbitBorderPane = new BorderPane();

    /**
     * toolBar
     */
    private ToolBar toolBar = new ToolBar();

    /**
     * save and cancel button
     */
    private HBox save_cancel;
    Button save = new Button("Save");
    Button cancel = new Button("Cancel");

    /**
     * text fields
     */
    public TextField name;
    public TextField math;
    public TextField say;
    public TextField iq;
    public TextField hours;
    public TextField wage;
    public TextField contracts;
    public TextField payPerContract;
    public TextField carrots;

    /**
     * controller
     */
    public Control control;

    /**
     * starts the gui
     *
     * @param primaryStage
     * @throws Exception
     */
    @Override
    public void start(Stage primaryStage) throws Exception {
        control = new Control();
        control.createHobbit("Bilbo Baggins", "1+1=10", "'I will give you a name and I shall call you Sting'", 10);
        control.createContract("John Doe", "5 / 1 = 5", "I'll work for contract", "13", 100, 1.5);
        control.createHourly("Fred Jones", "2 * 5 = 10", "I'll work for money", "50", 75, .2);
        //Save / Cancel
        save_cancel = new HBox(5);
        save_cancel.getChildren().addAll(save, cancel);
        save_cancel.setAlignment(Pos.BOTTOM_RIGHT);

        //Border Pane
        baseBorderPane.setTop(toolBar);
        baseBorderPane.setPadding(new Insets(10, 10, 10, 10));

        //Primary Stage
        Scene scene = new Scene(baseBorderPane, 500, 300);
        primaryStage.setTitle("PolyAbstraHeritance");
        primaryStage.setScene(scene);
        primaryStage.show();

        setFormButtonHandler(toolBar.hourly);
        setFormButtonHandler(toolBar.contract);
        setFormButtonHandler(toolBar.hobbit);

        setComboHandler(toolBar.comboBox);

        save.setOnAction(event -> delegator.save(this));
        cancel.setOnAction(this::clear);
    }


    /**
     * clears the text fields
     * @param event
     */
    private void clear(ActionEvent event) {
        if(name!=null)name.clear();
        if(math!=null)math.clear();
        if(say!=null)say.clear();
        if(iq!=null)iq.clear();
        if(hours!=null)hours.clear();
        if(wage!=null)wage.clear();
        if(contracts!=null)contracts.clear();
        if(payPerContract!=null)payPerContract.clear();
        if(carrots!=null)carrots.clear();
    }

    /**
     * handler for the combobox
     * @param box
     */
    private void setComboHandler(ComboBox box) {
        box.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                Text text = new Text("NULL");
                if (toolBar.comboBox.getValue() == "Math") {
                    text = new Text((control.getMath()));
                }
                if (toolBar.comboBox.getValue() == "Income") {
                    text = new Text((control.getIncome()));
                }
                if (toolBar.comboBox.getValue() == "Hours") {
                    text = new Text((control.getHours()));
                }
                if (toolBar.comboBox.getValue() == "IQ") {
                    text = new Text((control.getIQ()));
                }
                if (toolBar.comboBox.getValue() == "Say") {
                    text = new Text((control.getSay()));
                }
                if (toolBar.comboBox.getValue() == "Carrots") {
                    text = new Text(control.getCarrots());
                }
                if (toolBar.comboBox.getValue() == "Contracts") {
                    text = new Text(control.getContracts());
                }

                baseBorderPane.setCenter(text);
            }
        });
    }

    /**
     * handler for the 3 toolbar buttons
     * @param button
     */
    private void setFormButtonHandler(Button button) {
        button.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                if (event.getSource().equals(toolBar.hourly)) {
                    HBox hourlyCenter = new HBox();
                    VBox hourlyText = new VBox();
                    VBox hourlyField = new VBox();

                    hourlyText.getChildren().addAll(new Label("Name:"), new Label("Math:"),
                            new Label("Say:"), new Label("IQ:"),
                            new Label("Hours:"), new Label("Wage:"));
                    hourlyText.setPadding(new Insets(10, 10, 5, 10));
                    hourlyText.setSpacing(15);

                    name = new TextField();
                    math = new TextField();
                    say = new TextField();
                    iq = new TextField();
                    hours = new TextField();
                    wage = new TextField();

                    hourlyField.getChildren().addAll(name, math, say, iq, hours, wage);
                    hourlyField.setPadding(new Insets(5, 10, 5, 10));
                    hourlyField.setSpacing(5);

                    hourlyCenter.getChildren().addAll(hourlyText, hourlyField);
                    hourlyCenter.setAlignment(Pos.CENTER_LEFT);

                    hourlyBorderPane.setBottom(save_cancel);
                    hourlyBorderPane.setCenter(hourlyCenter);

                    baseBorderPane.setCenter(hourlyBorderPane);

                    delegator = delegateHourly;
                }

                if (event.getSource().equals(toolBar.contract)) {
                    HBox contractCenter = new HBox();
                    VBox contractText = new VBox();
                    VBox contractField = new VBox();

                    contractText.getChildren().addAll(new Label("Name:"), new Label("Math:"),
                            new Label("Say:"), new Label("IQ:"),
                            new Label("Contracts:"), new Label("Pay Per Contract:"));
                    contractText.setPadding(new Insets(10, 10, 5, 10));
                    contractText.setSpacing(15);

                    name = new TextField();
                    math = new TextField();
                    say = new TextField();
                    iq = new TextField();
                    contracts = new TextField();
                    payPerContract = new TextField();

                    contractField.getChildren().addAll(name, math, say, iq, contracts, payPerContract);
                    contractField.setPadding(new Insets(5, 10, 5, 10));
                    contractField.setSpacing(5);

                    contractCenter.getChildren().addAll(contractText, contractField);
                    contractCenter.setAlignment(Pos.CENTER_LEFT);

                    contractBorderPane.setBottom(save_cancel);
                    contractBorderPane.setCenter(contractCenter);

                    baseBorderPane.setCenter(contractBorderPane);

                    delegator = delegateContract;
                }

                if (event.getSource().equals(toolBar.hobbit)) {
                    HBox hobbitCenter = new HBox();
                    VBox hobbitText = new VBox();
                    VBox hobbitField = new VBox();

                    hobbitText.getChildren().addAll(new Label("Name:"), new Label("Math:"),
                            new Label("Say:"), new Label("Carrots:"));
                    hobbitText.setPadding(new Insets(10, 10, 5, 10));
                    hobbitText.setSpacing(15);

                    name = new TextField();
                    math = new TextField();
                    say = new TextField();
                    carrots = new TextField();

                    hobbitField.getChildren().addAll(name, math, say, carrots);
                    hobbitField.setPadding(new Insets(5, 10, 5, 10));
                    hobbitField.setSpacing(5);

                    hobbitCenter.getChildren().addAll(hobbitText, hobbitField);
                    hobbitCenter.setAlignment(Pos.CENTER_LEFT);

                    hobbitBorderPane.setBottom(save_cancel);
                    hobbitBorderPane.setCenter(hobbitCenter);

                    baseBorderPane.setCenter(hobbitBorderPane);

                    delegator = delegateHobbit;
                }
            }
        });
    }
}
