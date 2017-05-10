package cs2410.assn2;

/**
 * @author Philip Nelson
 * @version 1.0
 * @since 1/11/2017
 */

public class Fizzy
{
    private static int counter;

    /**
     * @return true of multiple of 3
     */
    private static boolean isFizz (int val) //check if a multiple of 3
    {
        return (val%3 == 0) ? true : false;
    }

    /**
     * @param val value to be checked
     * @return true if multiple of 5
     */
    private static boolean isBuzz(int val) //check if a multiple of 5
    {
        return (val%5 == 0) ? true : false;
    }

    /**
     * @param args command line arguments
     */
    public static void main(String[] args)
    {
        for(counter = 0; counter <= 100; ++counter)
        {
            if(isFizz(counter))
                System.out.print("Fizz");
            if(isBuzz(counter))
                System.out.print("Buzz");
            if(!isFizz(counter) && !isBuzz(counter))
            {
                System.out.print(counter);
            }
            System.out.println();
        }
    }
}
