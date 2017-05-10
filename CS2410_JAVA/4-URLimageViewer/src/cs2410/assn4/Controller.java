package cs2410.assn4;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by philip_nelson on 2/13/17.
 */
public class Controller {

    /**
     * reads in URLs from file
     * initializes currImage to 0
     */
    public Controller() {
        Scanner fileInput = null;
        try {
            fileInput = new Scanner(new FileReader(fileName));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }
        images = new ArrayList<String>();

        while (fileInput.hasNext()) {
            String url = fileInput.next();
            images.add(url);
        }

        currImage = 0;
    }

    /**
     * location of images file
     */
    private static final String fileName = "./data/images.data";

    /**
     * current image in array
     */

    private int currImage;
    /**
     * list of image URLs
     */
    ArrayList<String> images;

    /**
     * gets the next image to be displayed
     *
     * @return URL of next image
     */
    public String getNext() {
        if (currImage == images.size() - 1)
            currImage = 0;
        else ++currImage;
        return images.get(currImage);
    }

    /**
     * gets the previous image to be displayed
     *
     * @return URL of previous image
     */
    public String getPrev() {
        if (currImage == 0)
            currImage = images.size() - 1;
        else --currImage;
        return images.get(currImage);
    }

    /**
     * gets the current image to be displayed
     *
     * @return URL of current image
     */
    public String getCurr() {
        if(images.isEmpty())
            return "file:data/notfound.jpg";
        return images.get(currImage);
    }

    /**
     * add a URL to the list of images
     *
     * @param url the new URL to be added
     */
    public void add(String url) {
        images.add(currImage + 1, url);
    }

    /**
     * delets the current image from the list
     */
    public void deleteCurr() {
        images.remove(currImage);
        --currImage;
    }

    /**
     * get the number of images in the viewer
     * @return number of images in the viewer
     */
    public int size() {
        return images.size();
    }

    /**
     * called on close
     * prints array to file
     */
    public void quit()
    {
        PrintWriter fileOutput = null;
        try {
            fileOutput = new PrintWriter(new FileOutputStream(fileName));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        for(String s:images)
            fileOutput.println(s);
        fileOutput.close();
    }
}