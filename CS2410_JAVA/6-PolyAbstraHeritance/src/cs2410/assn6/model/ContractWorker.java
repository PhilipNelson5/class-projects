package cs2410.assn6.model;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class ContractWorker extends Smarty{
    /**
     * number of contracts
     */
    private double m_contracts;
    /**
     * pay per contract
     */
    private double m_payPerContract;

    /**
     * gets person type
     * @return person type
     */
    @Override
    public String getPersonType() {
        return "Contract";
    }

    /**
     * calculates income
     * @return income
     */
    @Override
    public double getIncome() {
        return m_contracts * m_payPerContract;
    }

    /**
     * contract worker constructor
     * @param name contract worker name
     * @param math contract worker math
     * @param say  contract worker say
     * @param iq   contract worker iq
     * @param contracts      contract worker contracts
     * @param payPerContract contract worker pay per contract
     */
    public ContractWorker(String name, String math, String say, String iq, double contracts, double payPerContract) {
        m_name = name;
        m_math = math;
        m_say = say;
        m_iq = iq;
        m_contracts = contracts;
        m_payPerContract = payPerContract;
    }

    /**
     * gets completed contracts
     * @return number of completed contracts
     */
    public double getContractsCompleted() {
        return m_contracts;
    }
}
