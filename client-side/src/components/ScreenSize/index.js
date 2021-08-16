import { useEffect, useState } from "react"

// Via: https://github.com/MarkMoretto/react-components/blob/main/dev-utils/currentscreensize-nextjs.js

const checkWindow = () => typeof window !== "undefined"


/**
 * Is there a `window` element in the DOM?
 * @type {Boolean}
*/
const getWindowDimensions = () => {

    const _width = checkWindow() ? window.innerWidth : null;
    const _height = checkWindow() ? window.innerHeight : null;
    return {
        _width,
        _height,
    };
}

/**
 * Using React hooks to output the current inner dimensions for a user's browser
 * @summary This is meant to work with Next.js. This returns a drop-in component that returns a basic <div> element.
 * @example -
 *   <ScreenSize />
 * @exports ScreenSize
*/
const useWindowDimensions = () => {
    const [hasWindow, setHasWindow] = useState(checkWindow());
    const [windowDimensions, setWindowDimensions] = useState(getWindowDimensions());

    useEffect(() => {
        setHasWindow(checkWindow())

        if (hasWindow) {
            function handleResize() {
                setWindowDimensions(getWindowDimensions());
            }

            window.addEventListener("resize", handleResize);
            return () => window.removeEventListener("resize", handleResize);
        }

    }, [hasWindow]);

    return windowDimensions;
}


const ScreenSize = () => {
    const { width, height } = useWindowDimensions();
    return (
        <div>Current window dimensions: {width} X {height}</div>
    );
}


export default ScreenSize
