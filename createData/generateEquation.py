import random

def generateFormula(complexity, symbol_count = 0):
# complexity near 0 = low, near 1 = high
#At beginning, I want: prob_of_equation = 0.4, prob_of_expression = 0.4, prob_of_negative = 0.2, prob_of_number = 0.3, prob_of_brackets = 0.2   
#By the end, I want: prob_of_equation = 0.6, prob_of_expression = 0.6, prob_of_negative = 0.3, prob_of_number = 0.5, prob_of_brackets = 0.4
#These equations assume linear function within max and min bounds
    prob_of_equation = 0.4 + complexity * 0.2
    prob_of_expression = 0.4 + complexity * 0.2
    prob_of_negative = 0.2 + complexity * 0.1
    prob_of_number = 0.3 + complexity * 0.2
    prob_of_brackets = 0.2 + complexity * 0.2

    #THIS IS TO CREATE NO EQUATIONS 
    prob_of_equation = 0.0
    if random.random() < prob_of_equation:
        expr1,symbol_count = generateExpression(prob_of_expression, prob_of_negative, prob_of_number, prob_of_brackets, symbol_count)
        expr2,symbol_count = generateExpression(prob_of_expression, prob_of_negative, prob_of_number, prob_of_brackets, symbol_count)
        symbol_count+=1
        return(expr1 + "= " + expr2[:-1], symbol_count)
    else:
        out_tuple = generateExpression(prob_of_expression, prob_of_negative, prob_of_number, prob_of_brackets, symbol_count)
        return out_tuple[0][:-1], out_tuple[1]
        

def generateExpression(prob_of_expression, prob_of_negative, prob_of_number, prob_of_brackets, symbol_count):
    if random.random() < prob_of_expression:
        new_prob_of_expression = prob_of_expression *0.9 # In order to avoid equations that are too long
        expr1,symbol_count = generateExpression(new_prob_of_expression, prob_of_negative, prob_of_number, prob_of_brackets, symbol_count)
        expr2,symbol_count = generateAtomic(prob_of_negative, prob_of_number, symbol_count)
        return generateOperator(prob_of_brackets, expr1, expr2, symbol_count)
    else:
        return generateAtomic(prob_of_negative, prob_of_number, symbol_count)

    

def generateAtomic(prob_of_negative, prob_of_number, symbol_count):
    out_string = ""
    if random.random() < prob_of_negative:
        out_string = "- "
        symbol_count += 1
    if random.random() < prob_of_number:
        out_string = out_string + random.choice(["a ","b ","c ","x ","y ","z "])
        symbol_count += 1
    else:
        my_number = str(random.randint(0,99))
        if len(my_number)==1:
            symbol_count += 1
        else:
            my_number = " ".join(my_number)
            symbol_count+=2
        out_string = out_string + my_number + " " 
    return out_string, symbol_count


def generateOperator(prob_of_brackets, expr1, expr2, symbol_count):

    # operator_list = ["+","-","*","/","^"]
    operator_list = ["+","-","*","^"]#DONT FORGET TO SWAP THIS BACK TO ABOVE

    my_operator = random.choice(operator_list)
    if my_operator== "+":
        symbol_count+=1
        output = expr1 +"+ "+expr2
    elif my_operator== "-":
        symbol_count+=1
        output = expr1 +"- "+expr2
    elif my_operator== "*":
        symbol_count+=1
        output = expr1 + "\\times " + expr2
    elif my_operator== "/":
        symbol_count+=5
        output = "\\frac { "+expr1+"} { "+expr2+"} "
    elif my_operator== "^":
        #first detect if needs brackets ie if its just a or a+2 (len 2 is one character with space), 
        #otherwise impose brackets but need to check if brackets already exist
        #always put expr2 in curly brackets
        if len(expr1) == 2: # this does not account for the case of 2 digit numbers
            symbol_count+=3
            output = expr1 + "^ { " + expr2 + "} "
        else:
            #check if brackets already exist
            expr1_as_list = expr1.split()
            if expr1_as_list[0] == "\\left(" and expr1_as_list[1] == "\\right)":
                symbol_count+=3
                output = expr1 + "^ { " + expr2 + "} "
            else:
                #this is when expr1 has no brackets, put brackets around expr1
                symbol_count+=5
                output = "\\left( "+ expr1 +"\\right) " + "^ { " + expr2 + "} "
    if random.random() < prob_of_brackets:
        symbol_count+=2
        output = "\\left( "+ output +"\\right) "
    return output, symbol_count