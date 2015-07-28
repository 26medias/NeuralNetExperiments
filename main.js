var _			= require("underscore");
var toolset		= require("toolset");
var neuralnet	= require("./neuralnet");

/*
// XOR Example
// Topology
var nn = new neuralnet({
	learning:	{
		rate:	0.2
	},
	topology: {
		input:	2,
		hidden:	[2],
		output:	1
	},
	error: {
		target:	0.1
	}
});

nn.trainingData([
	[[1,0],[1]],
	[[0,1],[1]],
	[[1,1],[0]],
	[[0,0],[0]]
]);
*/


// Theoretical problem
// Topology
var nn = new neuralnet({
	filename:	"nn.net",
	learning:	{
		rate:	0.2
	},
	topology: {
		input:	5,
		hidden:	[2],
		output:	2
	},
	error: {
		target:	0.1
	}
}, function() {
	nn.trainingData([
		[[1,1,1,0,0],[1,0]],
		[[0,0,1,1,1],[0,1]],
		[[1,0,1,1,1],[0,1]],
		[[1,1,1,0,1],[1,0]],
		[[1,0,0,0,1],[1,1]],
		[[0,0,0,0,0],[0,0]],
		[[0,1,1,1,0],[0,0]],
		[[1,1,1,1,1],[1,1]]
	]);
	
	nn.evaluate(10000);
	nn.train();
	
	/*
	nn.save("nn.net", function() {
		toolset.log("Status", "Net saved!");
	});
	*/
});