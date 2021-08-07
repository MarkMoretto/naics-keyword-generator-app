
/* 
    Supported elements: xlinkActuate xlinkArcrole xlinkHref xlinkRole xlinkShow xlinkTitle xlinkType xmlns xmlnsXlink
    https://reactjs.org/docs/dom-elements.html
*/


// import styles from "./Logo.module.css"

const LogoBackground = ({logoWidth = 300, logoHeight = 300, backgroundColor = "#ffcf41", props}) => {
    return (
        <rect 
            id="logo-container"
            width={logoWidth}
            height={logoHeight}
            fill={backgroundColor}
            {...props}
        />        
    )
}

const BrandLogo = ({ logoWidth = 300, logoHeight = 300, scaleRatio = 0.3, translateCoords = {X: 15, y: 5} }) => {

    return (
        <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink"
            width={`${logoWidth}`}
            height={`${logoHeight}`}
            viewBox={`0 0 ${logoWidth} ${logoHeight}`}
        >
            <g transform={`translate(${translateCoords.X}, ${translateCoords.y}) scale(${scaleRatio})`}>
                <defs>                 
                    <polyline 
                        id = "logo-lhs"
                        fill="black"
                        points="155,245
                                112,125
                                90,240
                                70,240
                                102,60
                                112,60
                                155,190"
                    />
                </defs>

                <LogoBackground id="logo-container" />               
                <use xlinkHref="#logo-lhs" />
                <use xlinkHref="#logo-lhs" transform="translate(310,0) scale(-1, 1)" />
            </g>
        </svg>
    )
}

export default BrandLogo