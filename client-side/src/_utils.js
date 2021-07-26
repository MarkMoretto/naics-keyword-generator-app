
/**
 * Curried fetch with JSON parsing.
 *
 * @template {Record<PropertyKey, unknown>} ResponseType
 * @param {RequestInit} init
 */
 const fetchJSON = init =>
    /**
     * @param {RequestInfo} input
     * @returns {() => Promise<ResponseType>}
     */
    input => () => fetch(input, init).then(response => response.json());

export default fetchJSON