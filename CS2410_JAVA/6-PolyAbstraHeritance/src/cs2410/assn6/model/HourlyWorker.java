package cs2410.assn6.model;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class HourlyWorker extends Smarty {
    /**
     * hours worked
     */
    private double m_hours;
    /**
     * hourly wage
     */
    private double m_wage;

    /**
     * gets person type
     * @return person type
     */
    @Override
    public String getPersonType() {
        return "Hourly";
    }

    /**
     * calculates hourly worker income
     * @return hourly worker income
     */
    @Override
    public double getIncome() {
        return m_hours * m_wage;
    }

    /**
     * hourly worker constructor
     * @param name  hourly worker name
     * @param math  hourly worker math
     * @param say   hourly worker say
     * @param iq    hourly worker iq
     * @param hours hourly worker hours
     * @param wage  hourly worker wage
     */
    public HourlyWorker(String name, String math, String say, String iq, double hours, double wage) {
        m_name = name;
        m_math = math;
        m_say = say;
        m_iq = iq;
        m_hours = hours;
        m_wage = wage;
    }

    /**
     * gets hours worked
     * @return hours worked
     */
    public double getHoursWorked() {
        return m_hours;
    }

}
