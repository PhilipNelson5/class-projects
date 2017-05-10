package cs2410.assn6.model;

/**
 * Created by philip_nelson on 3/28/17.
 */
public interface Simpleton extends PersonType{
    /**
     * gets the name string
     * @return name string
     */
    String getName();

    /**
     * gets the math string
     * @return math string
     */
    String doMath();

    /**
     * gets the say strings
     * @return say string
     */
    String saySometing();
}
