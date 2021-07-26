import {  useEffect } from "react"
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

// const getJSON = fetchJSON({ method: "GET" });
// business-related data solutions


/**
 * useEffect async helper.
 * @see https://stackoverflow.com/questions/53949393/cant-perform-a-react-state-update-on-an-unmounted-component
*/
export const useAsync = (asyncFunc, onSuccess) => {
    useEffect(() => {

        let isActive = true;
        
        asyncFunc().then(data => {
            if (isActive) onSuccess(data);
        });

        return () => { isActive = false };

    }, [asyncFunc, onSuccess]);
}

export default fetchJSON