package cs2410.assn6.control;

import cs2410.assn6.model.*;

import java.util.ArrayList;

/**
 * Created by philip_nelson on 3/28/17.
 */
public class Control {
    /**
     * list of workers
     */
    private ArrayList<Simpleton> workers;

    /**
     * constructor for controller
     */
    public Control() {
        workers = new ArrayList<Simpleton>();
    }

    /**
     * creates new hourly worker and adds to workers list
     * @param name  hourly worker name
     * @param math  hourly worker math
     * @param say   hourly worker say
     * @param iq    hourly worker iq
     * @param hours hourly worker hours
     * @param wage  hourly worker wage
     */
    public void createHourly(String name, String math, String say, String iq, double hours, double wage) {
        workers.add(new HourlyWorker(name, math, say, iq, hours, wage));
    }

    /**
     * creates new contract worker and adds to workers list
     * @param name contract worker name
     * @param math contract worker math
     * @param say  contract worker say
     * @param iq   contract worker iq
     * @param contracts      contract worker contracts
     * @param payPerContract contract worker pay per contract
     */
    public void createContract(String name, String math, String say, String iq, double contracts, double payPerContract) {
        workers.add(new ContractWorker(name, math, say, iq, contracts, payPerContract));
    }

    /**
     * creates new hobbit and adds to workers list
     * @param name    hobbit name
     * @param math    hobbit math
     * @param say     hobbit say
     * @param carrots hobbit carrots picked
     */
    public void createHobbit(String name, String math, String say, int carrots) {
        workers.add(new Hobbit(name, math, say, carrots));
    }

    /**
     * gets math strings
     * @return string with math strings
     */
    public String getMath() {
        StringBuffer math = new StringBuffer();
        for (Simpleton s : workers) {
            math.append(s.getName() + ", ");
            math.append(s.getPersonType() + ", ");
            math.append(s.doMath() + '\n');
        }
        return math.toString();
    }

    /**
     * gets income strings
     * @return string with income strings
     */
    public String getIncome() {
        StringBuffer income = new StringBuffer();
        for (Simpleton s : workers) {
            if (s instanceof HourlyWorker) {
                income.append(s.getName() + ", ");
                income.append(s.getPersonType() + ", ");
                income.append("made " + ((HourlyWorker) s).getIncome() + " moneys" + '\n');
            }
            if (s instanceof ContractWorker) {
                income.append(s.getName() + ", ");
                income.append(s.getPersonType() + ", ");
                income.append("made " + ((ContractWorker) s).getIncome() + " moneys" + '\n');
            }
        }
        return income.toString();
    }

    /**
     * gets hours strings
     * @return string with hours strings
     */
    public String getHours() {
        StringBuffer hours = new StringBuffer();
        for (Simpleton s : workers) {
            if (s instanceof HourlyWorker) {
                hours.append(s.getName() + ", ");
                hours.append(s.getPersonType() + ", ");
                hours.append("worked " + ((HourlyWorker) s).getHoursWorked() + " hours." + '\n');
            }
        }
        return hours.toString();
    }

    /**
     * gets iq strings
     * @return string with iq strings
     */
    public String getIQ() {
        StringBuffer iq = new StringBuffer();
        for (Simpleton s : workers) {
            if (s instanceof Smarty) {
                iq.append(s.getName() + ", ");
                iq.append(s.getPersonType() + ", ");
                iq.append("IQ is " + ((Smarty) s).getIQ() + '\n');
            }
        }
        return iq.toString();
    }

    /**
     * gets say strings
     * @return string with say strings
     */
    public String getSay() {
        StringBuffer say = new StringBuffer();
        for (Simpleton s : workers) {
            say.append(s.getName() + ", ");
            say.append(s.getPersonType() + ", ");
            say.append(s.saySometing() + '\n');
        }
        return say.toString();
    }

    /**
     * gets carrot strings
     * @return string with carrot strings
     */
    public String getCarrots() {
        StringBuffer carrots = new StringBuffer();
        for (Simpleton s : workers) {
            if (s instanceof Hobbit) {
                carrots.append(s.getName() + ", ");
                carrots.append(s.getPersonType() + ", ");
                carrots.append("Picked " + ((Hobbit) s).getCarrotsPicked() + " carrots!\n");
            }
        }
        return carrots.toString();
    }

    /**
     * gets contract strings
     * @return string with contact strings
     */
    public String getContracts() {
        StringBuffer contracts = new StringBuffer();
        for (Simpleton s : workers) {
            if (s instanceof ContractWorker) {
                contracts.append(s.getName() + ", ");
                contracts.append(s.getPersonType() + ", ");
                contracts.append("has " + ((ContractWorker) s).getContractsCompleted() + " contracts." + '\n');
            }
        }
        return contracts.toString();
    }
}
