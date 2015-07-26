var _			= require("underscore");
var toolset		= require("toolset");
var neuralnet	= require("./neuralnet");

// Topology
var nn = new neuralnet({
	learning:	{
		rate:	0.2
	},
	topology: {
		input:	2,
		hidden:	[2],
		output:	1
	}
});

nn.trainingData([
	[[1,0],[1]],
	[[0,1],[1]],
	[[1,1],[0]],
	[[0,0],[0]]
]);

console.log("No training");
nn.evaluate();

console.log("10 training");
nn.train(10);
nn.evaluate();

console.log("1000 training");
nn.train(1000);
nn.evaluate();

console.log("10000 training");
nn.train(10000);
nn.evaluate();

console.log("100000 training");
nn.train(100000);
nn.evaluate();