
// Grid helper
// TODO: Possibly add prop for number of columns.
// import PropTypes from "prop-types"

import styles from "./Griddy.module.css"

const Griddy = ({ itemArray, numberColumns = 4, classname = ""}) => {

    const colStyle = `col-${numberColumns}`

    return (
        <div className={classname}>
            <ul className={`${styles["grid-list"]} ${numberColumns ? styles[colStyle] : ""}`}>
            {Object.entries(itemArray).length > 0 
            && Object.values(itemArray)[0].map(item => {
                return (
                    <li id={item}>{item}</li>
                    )
                })
            }
            </ul>
        </div>          
    )
}

// Griddy.defaultProps  = {
//     itemArray: PropTypes.array,
//     numberColumns: 4,
//     classname: PropTypes.string,
// }

// Griddy.propTypes = {
//     itemArray: PropTypes.array,
//     numberColumns: PropTypes.number,
//     classname: PropTypes.string,
// }



export default Griddy
