import cowsay
import random

compliment_list = ["That was a moo-ving presentation! You really nailed it.",
	"Your dedication to this project is truly udderly amazing.",
	"I can always milk your brain for great ideas. Thanks for your insights!",
	"You're not just good, you're legen-dairy!",
	"Thanks for cow-ering all the details in that report; it was super thorough.",
	"You always cream the competition with your problem-solving skills.",
	"When it comes to teamwork, you're the cream of the crop!"
]

def cow_print(method, one_norm, T, Q):
	my_compliment = random.choice(compliment_list)

	cow_string = f"""Use the {method} LCU:\n   - one-norm: {one_norm:.2f}\n   - T-gates: {T:.2e}\n   - Qubits: {Q:.0f}\n\n{my_compliment}"""

	print(cowsay.get_output_string('cow', cow_string))
