package cs2410.assn3.command;

import cs2410.assn3.directory.Directory;

import java.util.Scanner;

/**
 * Created by Philip Nelson on 1/27/2017.
 */
public class CommandDirectory {

    /**
     * main method. controls flow of command line directory
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        Directory cmdir = new Directory();

        Scanner input = new Scanner(System.in);

        while (true) {
            menu();
            int choice = input.nextInt();
            switch (choice) {
                case 1:
                    System.out.println(cmdir.toString());
                    break;
                case 2:
                    cmdir.addStudent();
                    break;
                case 3:
                    System.out.printf("%s %.2f\n\n", "Average student age is: ", cmdir.getAve());
                    break;
                case 4:
                    System.exit(1);
                    break;
            }
        }
    }

    /**
     * displays the menu
     */
    public static void menu() {
        System.out.println("Menu:\n" +
                "1: List directory\n" +
                "2: Enter new student\n" +
                "3: Display average student age\n" +
                "4: Quit");
    }
}
