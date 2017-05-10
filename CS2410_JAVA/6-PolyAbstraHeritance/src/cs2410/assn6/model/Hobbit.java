package cs2410.assn6.model;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class Hobbit implements Simpleton, PersonType {
    /**
     * hobbit name
     */
    String m_name;
    /**
     * hobbit name
     */
    String m_math;
    /**
     * hobbit say
     */
    String m_say;
    /**
     * hobbit carrots
     */
    int m_carrots;

    /**
     * hobbit constructor
     * @param name    hobbit name
     * @param math    hobbit math
     * @param say     hobbit say
     * @param carrots hobbit carrots picked
     */
    public Hobbit(String name, String math, String say, int carrots){
        m_name = name;
        m_math = math;
        m_say = say;
        m_carrots = carrots;
    }

    /**
     * gets hobbit name
     * @return hobbit name
     */
    @Override
    public String getName() {
        return m_name;
    }

    /**
     * gets hobbit math
     * @return hobbit math
     */
    @Override
    public String doMath() {
        return m_math;
    }

    /**
     * gets hobbit say
     * @return hobbit say
     */
    @Override
    public String saySometing() {
        return m_say;
    }

    /**
     * gets person type
     * @return hobbit type
     */
    @Override
    public String getPersonType() {
        return "Hobbit";
    }

    /**
     * gets hobbit carrots
     * @return hobbit carrots
     */
    public int getCarrotsPicked()
    {
        return m_carrots;
    }
}
