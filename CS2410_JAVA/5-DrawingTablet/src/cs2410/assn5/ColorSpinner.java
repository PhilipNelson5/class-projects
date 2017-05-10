package cs2410.assn5;

import javafx.scene.paint.Color;
import javafx.scene.shape.Path;
import javafx.scene.shape.Shape;

import java.util.Random;

/**
 * Created by philip_nelson on 3/2/17.
 */
public class ColorSpinner implements Runnable {
    /**
     * used for termination of the thread
     */
    boolean running;

    /**
     * the shape to have its color changed
     */
    Shape shape;

    /**
     * random number generator
     */
    private Random rand = new Random();

    /**
     * constructor
     *
     * @param s the shape of which will have its color changed by the thread
     */
    ColorSpinner(Shape s) {
        shape = s;
    }

    /**
     * generates a random color object
     *
     * @return randomly generated color
     */
    private Color randColor() {
        float r = rand.nextFloat();
        float g = rand.nextFloat();
        float b = rand.nextFloat();
        Color randomColor = new Color(r, g, b, 1.0);
        return randomColor.brighter().brighter();
    }

    /**
     * sets the boolean running to false.
     * this will lead to termination of the thread
     */
    public void terminate() {
        running = false;
    }


    @Override
    /**
     * while the terminate has not been called the thread
     * will change the color of the given shape every 60ms
     */
    public void run() {
        running = true;
        boolean grow = true;
        int change = 4;
        int width = (int) shape.getStrokeWidth();
        while (running) {
            if (shape instanceof Path) {
                shape.setStroke(randColor());
            } else {
                shape.setFill(randColor());
//                shape.setStroke(randColor());
            }

            if (grow) {
                if (width >= 9) {
                    grow = false;
                    width -= change;
                } else
                    width += change;
            } else {
                if (width <= 1) {
                    grow = true;
                    width += change;
                } else
                    width -= change;
            }
            if (width < 1)
                shape.setStrokeWidth(0);
            else
                shape.setStrokeWidth(width);

            try {
                Thread.sleep((long) 60);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
