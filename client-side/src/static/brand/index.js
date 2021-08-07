
// ref: https://pganalyze.com/blog/building-svg-components-in-react
import styles from "./Logo.module.css"

const BrandLogo = props => {
    // Viewbox attrs {min-x min-y width height}
    const viewboxOffsetX = 0
    const viewboxOffsetY = 0
    // const windowWidth = 24 // "79.117477mm"
    // const windowHeight = 30 // "79.775368mm"

    const viewBoxWidth = 80
    const viewBoxHeight = 80
    const gTranslateX = -67.25
    const gTranslateY = -35.71
    
    return (
        <svg xmlns="http://www.w3.org/2000/svg" id="svg-logo" viewBox={`${viewboxOffsetX} ${viewboxOffsetY} ${viewBoxWidth} ${viewBoxHeight}`}>
            <g id="layer1" transform={`translate(${gTranslateX}, ${gTranslateY})`}>
                <g id="svg-logo-g" className={styles.gClass}>
                <path 
                    className={styles.path1}
                    id="path1105"
                    d="M 67.376062,35.842501 H 146.23377 V 115.35817 H 67.419606 Z"
                />
                <path
                    className={styles.path2}
                    id="path1103"
                    d="m 84.666667,99.21875 9.260416,-47.625 h 2.645834 L 107.15625,84.666667 117.73958,51.59375 h 2.64584 l 9.26041,47.625 h -5.29166 l -5.29167,-31.75 -11.90625,31.75 -11.90625,-31.75 -5.291667,31.75 z"
                />
                </g>
            </g>
        </svg>     
    )
}

export default BrandLogo