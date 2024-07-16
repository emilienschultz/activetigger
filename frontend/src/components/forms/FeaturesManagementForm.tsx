import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useAddFeature, useDeleteFeature } from '../../core/api';
import { useNotifications } from '../../core/notifications';
import { SchemeModel } from '../../types';

interface FeaturesManagementProps {
  availableFeatures: string[];
  projectSlug: string;
  reFetchProject: () => void;
}

/* Manage schemes
 * Select ; Delete ; Add
 */

// TODO
interface FeatureModel {
  to_define: string;
}

export const FeaturesManagement: FC<FeaturesManagementProps> = ({
  availableFeatures,
  projectSlug,
  reFetchProject,
}) => {
  // hooks to use the objets
  const addFeature = useAddFeature(projectSlug);
  const { register, handleSubmit } = useForm<FeatureModel>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const deleteFeature = useDeleteFeature(projectSlug, current_scheme);

  // state for displaying the new scheme menu
  const [showCreateNewFeature, setShowCreateNewFeature] = useState(false);
  const handleAddIconClick = () => {
    setShowCreateNewFeature(!showCreateNewFeature);
  };

  // action to create the new scheme
  const createNewFeature: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      addFeature(formData.name);
      notify({ type: 'success', message: `Feature created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    reFetchProject();
    setShowCreateNewFeature(!showCreateNewFeature);
  };

  // action to delete feature
  const deleteSelectedFeature = async () => {
    //TODO: try catch and throw
    await deleteFeature();
    reFetchProject();
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <label className="form-label" htmlFor="existing-feature-selected">
          Select existing feature
        </label>
        <div>
          <select id="existing-feature-selected" className="form-select-lg col-3 mb-3">
            <option></option> {/*empty possibility*/}
            {availableFeatures.map((element) => (
              <option key={element} value={element}>
                {element}
              </option>
            ))}{' '}
          </select>
          <button onClick={deleteSelectedFeature} className="btn btn p-0">
            <MdOutlineDeleteOutline size={30} />
          </button>
          <button onClick={handleAddIconClick} className="btn p-0">
            <IoIosAddCircle size={30} />
          </button>
        </div>
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewFeature && (
            <div className="row">
              <form onSubmit={handleSubmit(createNewFeature)}>
                <div className="secondary-panel col-4">
                  <label className="form-label" htmlFor="newFeature">
                    New scheme
                  </label>
                  <select className="form-control" id="newFeature" {...register('name')}></select>
                  <button className="btn btn-primary btn-validation">Create</button>
                </div>
              </form>
            </div>
          )
        }
      </div>
    </div>
  );
};
