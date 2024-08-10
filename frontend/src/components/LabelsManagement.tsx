import { FC, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { FaPlusCircle } from 'react-icons/fa';
import { RiFindReplaceLine } from 'react-icons/ri';

import { useAddLabel, useDeleteLabel, useRenameLabel } from '../core/api';

interface LabelsManagementProps {
  projectName: string;
  currentScheme: string;
  availableLabels: string[];
  reFetchCurrentProject: any;
}

export const LabelsManagement: FC<LabelsManagementProps> = ({
  projectName,
  currentScheme,
  availableLabels,
  reFetchCurrentProject,
}) => {
  // hooks to manage labels
  const { addLabel } = useAddLabel(projectName, currentScheme);
  const { deleteLabel } = useDeleteLabel(projectName, currentScheme);
  const { renameLabel } = useRenameLabel(projectName, currentScheme);

  // manage label creation
  const [createLabelValue, setCreateLabelValue] = useState('');
  const handleCreateLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCreateLabelValue(event.target.value);
  };
  const createLabel = () => {
    addLabel(createLabelValue);
    if (reFetchCurrentProject) reFetchCurrentProject();
  };

  // manage label deletion
  const [deleteLabelValue, setDeleteLabelValue] = useState('');
  const handleDeleteLabelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setDeleteLabelValue(event.target.value);
  };
  const removeLabel = () => {
    deleteLabel(deleteLabelValue);
    if (reFetchCurrentProject) reFetchCurrentProject();
  };

  // manage label replacement
  const replaceLabel = () => {
    renameLabel(deleteLabelValue, createLabelValue);
    setCreateLabelValue('');
    if (reFetchCurrentProject) reFetchCurrentProject();
  };

  return (
    <div>
      <div className="d-flex align-items-center">
        <select id="delete-label" onChange={handleDeleteLabelChange}>
          {availableLabels.map((e, i) => (
            <option key={i}>{e}</option>
          ))}{' '}
        </select>
        <button onClick={removeLabel} className="btn btn p-0">
          <FaRegTrashAlt size={20} className="m-2" />
        </button>
      </div>
      <div className="d-flex align-items-center">
        <input
          type="text"
          id="new-label"
          value={createLabelValue}
          onChange={handleCreateLabelChange}
          placeholder="Enter new label"
        />
        <button onClick={createLabel} className="btn btn p-0">
          <FaPlusCircle size={20} className="m-2" />
        </button>
      </div>
      <div className="d-flex align-items-center">
        Replace selected label to the new one
        <button onClick={replaceLabel} className="btn btn p-0">
          <RiFindReplaceLine size={20} className="m-2" />
        </button>
      </div>
    </div>
  );
};
