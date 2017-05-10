package cs2410.assn6.model;

/**
 * Created by philip_nelson on 3/28/17.
 */
public abstract class Smarty implements Simpleton, PersonType {
    /**
     * name
     */
    String m_name;
    /**
     * math
     */
    String m_math;
    /**
     * say
     */
    String m_say;
    /**
     * IQ
     */
    String m_iq;

    /**
     * gets the name
     * @return name string
     */
    @Override
    public String getName() {
        return m_name;
    }

    /**
     * gets the math
     * @return math string
     */
    @Override
    public String doMath() {
        return m_math;
    }

    /**
     * gets the say
     * @return say string
     */
    @Override
    public String saySometing() {
        return m_say;
    }

    /**
     * gets the iq
     * @return iq string
     */
    public String getIQ() {
        return m_iq;
    }

    /**
     * gets the income
     * @return income string
     */
    abstract double getIncome();
}
