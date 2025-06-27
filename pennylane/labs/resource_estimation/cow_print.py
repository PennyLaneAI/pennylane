import cowsay
import random

compliment_list = ["You're in a super-moo-sition of cool and awesome!",
	"Your dedication is truly udderly amazing.",
	"You're simply the cream of the crop.",
	"I can always milk your brain for great ideas.",
	"You're not just good, you're legen-dairy!",
	"You always cream the competition with your problem-solving skills.",
	"You're simply moo-velous",
	"You're moo-sic to my ears",
	"You make every day moo-tiful",
	"I'm moo-ved by your talent",
	"When it comes to teamwork, you're the cream of the crop!"
]

def cow_print(method, one_norm, T, Q, params):
	my_compliment = random.choice(compliment_list)

	cow_string = f"""Use the {method} LCU with the hyperparameters {params}:\n   - one-norm: {one_norm:.2f}\n   - T-gates: {T:.2e}\n   - Qubits: {Q:.0f}\n\n{my_compliment}"""

	print(cowsay.get_output_string('cow', cow_string))
