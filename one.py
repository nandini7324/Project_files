import random
while True :
 print("😉-----number guessing game----😀")
 s=int(input("enter the start number :"))
 e=int(input("enter the end number:"))
 print(f"your range is {s} to {e} ")

 ran_no=random.randint(s,e)

 while True :
     x=int(input("Enter your guessing number"))
     if ran_no ==x :
      print("you guess the correct ✔")
      print("you got it 👍")
      break
     elif ran_no >x :
      print(f"the number greater than {x} ")
     elif ran_no <x :
      print(f"the number is less than {x} ")
 