package cs2410.assn3.directory;

/**
 * Created by Philip Nelson on 1/26/2017.
 */

public class Student {
    /**
     * Constructor: creates new student
     *
     * @param f First Name
     * @param l Last Name
     * @param a Age
     * @param p Phone Number
     */
    Student(String f, String l, int a, String p) {
        fname = f;
        lname = l;
        pnumber = p;
        age = a;
    }

    /**
     * First Name
     */
    private String fname;
    /**
     * Last Name
     */
    private String lname;
    /**
     * Age
     */
    private int age;
    /**
     * Phone Number
     */
    private String pnumber;

    /**
     * creates string of student object
     *
     * @return string with student information
     */
    public String toString() {
        String pn = "   " + String.format(String.valueOf(pnumber).replaceFirst("(\\d{3})(\\d{3})(\\d+)", "($1)-$2-$3"));
        return String.format("%-10s %-10s %3d", fname, lname, age) + pn
                + ("\n");
    }

    /**
     * gets age
     *
     * @return student age
     */
    public int getAge() {
        return age;
    }
}
