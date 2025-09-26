
import random
import numpy as np

def simulate_coin_tosses(num_tosses=10000):
    """
    simulates tossing a coin a specified number of 
    time and calculates the experimental probability
    of heads and tails.
    args: num_tosses(int) : the number of times to toss the coin.
    """
    head_count = 0
    tails_count = 0
    
    # loop for trpeated trials
    for _ in range(num_tosses):
        # use random.choice to simulate a single coin toss
        toss = random.choice(['heads','tails'])
        
        #track outcomes
        if toss == 'heads':
            head_count += 1
        else:
            tails_count += 1
        
        #compute probabilities
    prob_head = head_count / num_tosses
    prob_tails = tails_count / num_tosses
        
    print(f"___coin toss simulation({num_tosses} tosses)___")
    print(f"Head count: {head_count}")
    print(f"tails count : {tails_count}")
    print(f"Experimental probability of heads :{prob_head:4f}")
    print(f"Experimental probability of tails: {prob_tails:.4f}\n")

def simulate_dice_rolls_sum(num_trials=50000):
   """
   simulates rolling two dice and computes the probability 
   of getting a sum of 7.
   
   args:num_trials (int) : The number of times to roll the two dice.
   """
   sum_is_7_count = 0
   #loop for repeated trails
   for _ in range(num_trials):
        die1 = random.randint(1,6)
        die2 = random.randint(1,6)
        
        #track outcomes where the sum is 7
        if die1 + die2 ==7:
            sum_is_7_count += 1
            
   #compute probability
   probability = sum_is_7_count / num_trials
   
   print(f"--- two dice roll simulation({num_trials}trials)---")
   print(f"number of times sum was 7L{sum_is_7_count}")
   print(f"experimental probability of sum being 7:{probability:.4f}")
   print("(theoretical probability id 1/6 or approx.0.1667)\n")

def prob_getting_at_least_one_six(num_rolls=10,num_simulation = 50000):
    """
    estimate the probability of getting at least one 6 in a givn numeber of rolls of a fair die
    
    arg :
        num_rolls (int): the number of times the die is rolled in one trail
        num _simulation(int):the total numbe of trails to run>
    return :
        float: the estimated probability
        """
    successful_trials = 0
    # outer loop for each complete simulation(e.g , 10 rolls)
    for _ in range(num_simulation):
        found_a_six = False
        # inner loop for the individual rolls within a single trails
        for _ in range(num_rolls):
            roll = random.randint(1,6)
            if roll == 6:
                #Track trials where at least one '6" occurs
                found_a_six = True
                # optimization : once a 6 is found, we can stop this trails
                break
    if found_a_six:
        successful_trials += 1
   #calulate the proportion of successful trials
    return successful_trials  / num_simulation
def simulate_ball_draws(num_draws=1000):
    """
    simulartes drawing balls form a bag with replacement ot demonstrate
    conditional probability and bayes theorem for independent events. 
    a bag contain 5 red , 7 green , and 8 blue balls.
    """
    bag = ['red'] * 5 + ['green'] * 7 + ['blue'] * 8
    # total balls = 20

    #simulate all draws first
    draws = [random.choice(bag) for _ in range(num_draws)]

    #----part a :P(red | previous was blue)---
    # we need to count  pair    
    for i in range(1, num_draws):
        if draws[i-1] == 'blue' :
            previous_was_blue_count += 1
            if draws[i] == 'red':
                current_red_and_previous_blue_count += 1
            

    print(f"--- Ball Draw Simulation ({num_draws} draws) ---")
    if previous_was_blue_count > 0:
        prob_red_given_blue = current_red_and_previous_blue_count / previous_was_blue_count
        print(f"a.P(red | previous was blue): {prob_red_given_blue:.4f}")
    else:
        print("a.No instances of a blue ball being drawn previously.")
    #With replacement, the events are independent, so P(Red | Blue) = P(Red)
    theoretical_prob_red = 5/20
    print(f"  (Theoretical P(Red) is {theoretical_prob_red:.4f} as events are independent)\n")

    # --- Part b: Verify Bayes' Theorem ---
    # P(A|B) = [P(B|A) * P(A)] / P(B)
    # Let A = Current draw is Red
    # Let B = Previous draw was Blue
    print("b. Verifying Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B)")

    # P(A): Simple probability of drawing a red ball
    red_count = draws.count('red')
    prob_A = red_count / num_draws

    # P(B): Simple probability of drawing a blue ball
    # (We use the count of 'previous was blue' over all possible previous slots)
    prob_B = previous_was_blue_count / (num_draws - 1) if num_draws > 1 else 0

    # P(B|A): P(Previous was Blue | Current is Red)
    current_was_red_count = 0
    # The joint event (blue then red) is the same, so we re-use the counter
    for i in range(1, num_draws):
        if draws[i] == 'red':
            current_was_red_count += 1
    
    prob_B_given_A = 0
    if current_was_red_count > 0:
        prob_B_given_A = current_red_and_previous_blue_count / current_was_red_count

    print(f"   From simulation:")
    print(f"   LHS: P(A|B) [from part a] = {prob_red_given_blue:.4f}")
    print(f"   RHS Components:")
    print(f"     P(A) = P(Red) = {red_count}/{num_draws} = {prob_A:.4f}")
    print(f"     P(B) = P(Previous Blue) = {previous_was_blue_count}/{num_draws - 1} = {prob_B:.4f}")
    print(f"     P(B|A) = P(Previous Blue | Current Red) = {current_red_and_previous_blue_count}/{current_was_red_count} = {prob_B_given_A:.4f}")
    
    # Calculate the right side of Bayes' theorem and compare
    if prob_B > 0:
        bayes_rhs = (prob_B_given_A * prob_A) / prob_B
        print(f"\n   RHS Calculated: ({prob_B_given_A:.4f} * {prob_A:.4f}) / {prob_B:.4f} = {bayes_rhs:.4f}")
        print(f"   Result: The LHS ({prob_red_given_blue:.4f}) and RHS ({bayes_rhs:.4f}) are approximately equal.")
    else:
        print("\n   Cannot calculate RHS of Bayes' Thm because P(B) is zero.")
    print("") # Add a newline for better formatting


def simulate_discrete_random_variable(sample_size=1000):
    """
    Generates a sample from a discrete random variable and computes its
    empirical mean, variance, and standard deviation.
    Distribution: P(X=1)=0.25, P(X=2)=0.35, P(X=3)=0.4
    """
    outcomes = [1, 2, 3]
    probabilities = [0.25, 0.35, 0.40]

    # a. Use numpy.random.choice() to generate the sample.
    sample = np.random.choice(outcomes, size=sample_size, p=probabilities)

    # b. Use numpy methods to calculate mean, variance, and standard deviation.
    mean = np.mean(sample)
    variance = np.var(sample)
    std_dev = np.std(sample)

    print(f"--- Discrete Random Variable Simulation (sample size={sample_size}) ---")
    print(f"Empirical Mean: {mean:.4f}")
    print(f"Empirical Variance: {variance:.4f}")
    print(f"Empirical Standard Deviation: {std_dev:.4f}\n")

    # For comparison, let's calculate the theoretical values
    theoretical_mean = sum(o * p for o, p in zip(outcomes, probabilities))
    theoretical_variance = sum(((o - theoretical_mean)**2) * p for o, p in zip(outcomes, probabilities))
    print(f"Theoretical Mean: {theoretical_mean:.4f}")
    print(f"Theoretical Variance: {theoretical_variance:.4f}\n")

def main():
    """ mainfunction to run all simulation."""
    # 1a tossinf a coin 10,000 times
    simulate_coin_tosses(10000)
    #1b. rolling two dice for a sum of 7
    simulate_dice_rolls_sum(50000)
    #2.estimate probability of at least one "6" in 10 rolls
    estimated_probability = prob_getting_at_least_one_six(num_rolls=10, num_simulation=50000)
    print(f"---at least one '6' in 10 Rolls Simulation ---")
    # The theoretical probability is 1 - (5/6)^10
    theoretical_prob = 1 - (5/6)**10
    
    print(f"Estimated Probability: {estimated_probability:.4f}")
    print(f"Theoretical Probability: {theoretical_prob:.4f}")

    # 3.conditional probability and ball draws
    simulate_ball_draws(1000)
    # 4.discrete random variable simulation
    simulate_discrete_random_variable(1000)


if __name__ == "__main__":
    main()
    