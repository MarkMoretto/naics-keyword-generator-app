/* eslint no-unused-vars: 0 */


/**
 * Generate random number from [min,max)
 * @see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random
*/
export const getRandomInt = (min, max) => {
    // Handle missing arguments.
    max = max || null
    min = min || null
    try {
        // Determine if either param missing and assign accordingly.
        let _max = max === null ? min : max
        let _min = max === null ? 0 : min

        _min = Math.ceil(_min)
        _max = Math.floor(_max)

        return Math.floor(Math.random() * (_max - _min) + _min);
    } catch  (error) {
        console.error(error)
    }
}

