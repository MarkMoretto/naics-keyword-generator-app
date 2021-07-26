

function curry(fn, arity = fn.length){
    return ( function nextCurry(previousArguments) {
        return function curried(nextArgument){
            var args = [...previousArguments, nextArgument];

            if(args.length >= arity){
                return fn(...args);
            }else{
                return nextCurry(args);
            }
        };
    })([]);
}


// Example
function sum(number1, number2, number3){
    return number1 + number2 + number3;
}

let curriedSum = curry(sum);

curriedSum(1)(2)(3) // 6