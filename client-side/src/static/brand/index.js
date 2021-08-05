
// ref: https://pganalyze.com/blog/building-svg-components-in-react


const BrandLogo = () => {
    // Viewbox attrs {min-x min-y width height}
    const viewboxOffsetX = -5
    const viewboxOffsetY = 0
    const windowWidth = 24
    const windowHeight = 30

    const viewBoxWidth = windowWidth + Math.abs(viewboxOffsetX)
    const viewBoxHeight = windowHeight + Math.abs(viewboxOffsetY)

    return (
        <svg xmlns="http://www.w3.org/2000/svg" id="svg-logo" viewBox={`${viewboxOffsetX} ${viewboxOffsetY} ${viewBoxWidth} ${viewBoxHeight}`}>
            {/* <Translate {...body.pos}> */}
            <path
                id="logo-path"
                d="M16.5.8v10.5H9.8c-1.3 0-2.4-1.1-2.4-2.4V.8c.1-.4-.3-.8-.7-.8H0v30h6.7c.4 0 .8-.4.8-.8V18.4h6.7c1.3 0 2.4 1.1 2.4 2.4v8.4c0 .4.4.8.8.8H24V0h-6.7c-.4 0-.8.4-.8.8z"
            >
            </path>
            {/* </Translate> */}
        </svg>        
    )
}

export default BrandLogo