import { FC } from 'react';
import { useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useDeleteScheme } from '../../core/api';
import { useAppContext } from '../../core/context';
import { SchemeModel } from '../../types';

interface SchemesManagementProps {
  available_schemes: string[];
  projectSlug: string;
}

/* Manage schemes*/
export const SchemesManagement: FC<SchemesManagementProps> = ({
  available_schemes,
  projectSlug,
}) => {
  const {
    appContext: { current_scheme }, //destructuration of the object from hook
    setAppContext,
  } = useAppContext();

  // hook to get the api call
  const deleteScheme = useDeleteScheme(projectSlug, current_scheme);

  // put the current scheme in the context
  const handleChange = (event: any) => {
    setAppContext((state) => ({
      ...state,
      current_scheme: event.target.value,
    }));
    console.log(`Current scheme ${event.target.value}`);
  };

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const handleIconClick = () => {
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  return (
    <div>
      <label className="form-label" htmlFor="scheme-selected">
        Select current scheme
      </label>
      <div>
        <select id="scheme-selected" className="form-select-lg mb-3 col-3" onChange={handleChange}>
          <option></option> {/*empty possibility*/}
          {available_schemes.map((element) => (
            <option key={element} value={element}>
              {element}
            </option>
          ))}{' '}
        </select>
        <button onClick={deleteScheme} className="btn">
          <MdOutlineDeleteOutline size={30} />
        </button>
        <button onClick={handleIconClick} className="btn">
          <IoIosAddCircle size={30} />
        </button>
      </div>
      <div>{showCreateNewScheme && <CreateNewScheme />}</div>
    </div>
  );
};

/* New Scheme creation */
export const CreateNewScheme: FC = () => {
  const { register, handleSubmit } = useForm<SchemeModel>({});

  // action when form validated
  const onSubmit: SubmitHandler<SchemeModel> = async () => {};

  return (
    <div className="container-fluid">
      <div className="row">
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="secondary-panel col-4">
            <label className="form-label" htmlFor="scheme_name">
              New scheme
            </label>
            <input className="form-control" id="scheme_name" type="text" {...register('name')} />
            <button className="btn btn-primary btn-validation">Create</button>
          </div>
        </form>
      </div>
    </div>
  );
};
