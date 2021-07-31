
import DataList from "../DataList"

/**
 * 
*/

let start = 0
,   stepsize = 1

let numArr = [...Array(5)].map((_, i) => i)
const quantityOptions = [
    {id: 0, value: " "},
    {id: 1, value: "1"},
]

const InputField = props => {

    return (
        <>
            <label htmlFor="id_sample_size">No. of Words:</label>
            <input 
                id="id_sample_size"
                name="sample_size"
                type="text"
                placeholder="..."
                onFocus={(e) => e.target.placeholder = ""} 
                onBlur={(e) => e.target.placeholder = "..."}                        
            />
            <DataList />
        </>
    )
}

export default InputField
