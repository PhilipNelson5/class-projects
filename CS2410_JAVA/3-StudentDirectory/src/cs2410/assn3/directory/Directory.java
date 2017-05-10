package cs2410.assn3.directory;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by Philip Nelson on 1/26/2017.
 */

public class Directory {
    /**
     * Constructor: reads in existing information from directory file
     */
    public Directory() {
        Scanner fileInput = null;
        try {
            fileInput = new Scanner(new FileReader(fileName));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Existing directory file not found, creating new directory.");
            dir = new ArrayList<Student>();
            return;
        }
        dir = new ArrayList<Student>();

        while (fileInput.hasNext()) {
            String fn = fileInput.next();
            String ln = fileInput.next();
            int age = fileInput.nextInt();
            String pn = fileInput.next();
            dir.add(new Student(fn, ln, age, pn));
        }
    }

    /**
     * location of directory file
     */
    private static final String fileName = "./data/cs2410-directory.data";

    /**
     * list of students
     */
    ArrayList<Student> dir;

    /**
     * turns directory inro a string
     *
     * @return the directory as a string
     */
    public String toString() {
        String str = "";
        for (Student s : dir) {
            str += s.toString();
        }
        return str;
    }

    /**
     * ask user for information and create new student and add to directory
     */
    public void addStudent() {
        PrintWriter fileOutput = null;

        Scanner input = new Scanner(System.in);
        String fn, ln, pn;
        int age;
        System.out.print("Enter first name: ");
        fn = input.next();
        System.out.print("Enter last name: ");
        ln = input.next();
        System.out.print("Enter age: ");
        age = input.nextInt();
        System.out.print("Enter phone number (digits only): ");
        pn = input.next();

        dir.add(new Student(fn, ln, age, pn));

        try {
            fileOutput = new PrintWriter(new FileOutputStream(fileName, true)); //try changing to false
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        fileOutput.println(fn + " " + ln + " " + age + " " + pn);
        fileOutput.close();
        System.out.println("\nThe following student has been added to the directory:");
        System.out.print(fn + " " + ln + ", " + "age: " + age + ", " + "phone: ");
        System.out.printf(String.valueOf(pn).replaceFirst("(\\d{3})(\\d{3})(\\d+)", "($1)-$2-$3"));
        System.out.println('\n');

    }

    /**
     * create new student and add to directory
     *
     * @param fn  First Name
     * @param ln  Last Name
     * @param age Age
     * @param pn  Phone Number
     * @return Confirmation string of student information
     */
    public String addStudent(String fn, String ln, int age, String pn) {
        PrintWriter fileOutput = null;

        dir.add(new Student(fn, ln, age, pn));

        try {
            fileOutput = new PrintWriter(new FileOutputStream(fileName, true)); //try changing to false
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        fileOutput.println(fn + " " + ln + " " + age + " " + pn);
        fileOutput.close();

        return "The following student has been added to the directory:\n" +
                fn + " " + ln + ", " + "age: " + age + ", " + "phone: " +
                String.valueOf(pn).replaceFirst("(\\d{3})(\\d{3})(\\d+)", "($1)-$2-$3");
    }

    /**
     * Calculate average student age
     *
     * @return average age
     */
    public double getAve() {
        double sum = 0.0;

        for (Student s : dir) {
            sum += s.getAge();
        }

        double ave = sum / dir.size();
        return ave;
    }
}
