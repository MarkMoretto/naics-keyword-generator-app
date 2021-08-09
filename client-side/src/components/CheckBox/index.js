

import "./styles.css"

/**
 * CheckBox component
*/
const CheckBox = props => {
    return (
        <label className="checkbox">
            <span className="checkbox__input">
                <input type="checkbox" {...props} />
                <span className="checkbox__control">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                        <path
                            fill="none"
                            stroke-width="3"
                            d="M1.73 12.91l6.37 6.37L22.79 4.59"
                        />
                    </svg>
                </span>
            </span>
        </label>
    )
}

export default CheckBox