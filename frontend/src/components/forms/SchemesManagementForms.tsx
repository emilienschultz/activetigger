import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useAddScheme, useDeleteScheme } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { SchemeModel } from '../../types';

interface SchemesManagementProps {
  available_schemes: string[];
  projectSlug: string;
  reFetchProject: () => void;
}

/* Manage schemes*/
export const SchemesManagement: FC<SchemesManagementProps> = ({
  available_schemes,
  projectSlug,
  reFetchProject,
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
        <button
          onClick={async () => {
            //TODO: try catch and throw
            await deleteScheme();
            reFetchProject();
          }}
          className="btn"
        >
          <MdOutlineDeleteOutline size={30} />
        </button>
        <button onClick={handleIconClick} className="btn">
          <IoIosAddCircle size={30} />
        </button>
      </div>
      <div>
        {showCreateNewScheme && (
          <CreateNewScheme projectSlug={projectSlug} reFetchProject={reFetchProject} />
        )}
      </div>
    </div>
  );
};

interface CreateNewSchemeProps {
  projectSlug: string;
  reFetchProject: () => void;
}

/* New Scheme creation */
export const CreateNewScheme: FC<CreateNewSchemeProps> = ({ projectSlug, reFetchProject }) => {
  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({});
  const addScheme = useAddScheme(projectSlug);
  const { notify } = useNotifications();

  // action when form validated
  const onSubmit: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      addScheme(formData.name);
      notify({ type: 'success', message: 'Scheme created' });
      reFetchProject();
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
  };

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
